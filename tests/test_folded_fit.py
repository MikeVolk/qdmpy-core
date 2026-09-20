"""Unit tests for folded fitting (QEP-059: unified constraint interface).

All tests use synthetic FoldedODMR data fitted through an injectable
FitBackend (QEP-068) — no GPU or mocked pygpufit internals required.

With QEP-059, folded fits use absolute-GHz frequencies (D_ZFS + delta_f)
and return a standard FitResult. No FoldedFitResult subclass exists.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from qdmpy.constants import D_ZFS, GAMMA_NV
from qdmpy.exceptions import DependencyError
from qdmpy.fitting.backends import BackendFitOutput, with_forced_availability
from qdmpy.fitting.manager import FitManager
from qdmpy.fitting.result import FitResult
from qdmpy.odmr.folding import FoldedODMR, FoldingSettings
from qdmpy.settings import (
    FitSettings,
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
)
from qdmpy.testing import FakeFitBackend

# ── Settings ────────────────────────────────────────────────────────────────

MOCK_SETTINGS = QDMpySettings(
    fit=FitSettings(
        estimator="LSE",
        max_number_iterations=100,
        tolerance=1e-6,
    ),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.70,
            center_max=3.04,
            center_type="FREE",
            width_min=0.0001,
            width_max=0.020,
            width_type="FREE",
            contrast_min=0.0,
            contrast_max=2.0,
            contrast_type="FREE",
            offset_min=-0.5,
            offset_max=3.0,
            offset_type="FREE",
        )
    ),
)

# ── Helpers ──────────────────────────────────────────────────────────────────

NY, NX = 4, 4
N_DF = 20  # number of delta_f frequency points
DELTA_F = np.linspace(0.005, 0.060, N_DF)  # GHz Zeeman offsets


def _make_folded_odmr(n_pol: int = 2) -> FoldedODMR:
    """Build a minimal synthetic FoldedODMR object."""
    pol_labels = ["neg", "pos"][:n_pol]
    spec = np.ones((n_pol, NY, NX, N_DF), dtype=np.float32)

    folded_da = xr.DataArray(
        spec,
        dims=("polarity", "y", "x", "freq_idx"),
        coords={
            "polarity": pol_labels,
            "delta_f_ghz": ("freq_idx", DELTA_F),
        },
    )
    anti_da = xr.DataArray(
        np.zeros_like(spec),
        dims=("polarity", "y", "x", "freq_idx"),
        coords={
            "polarity": pol_labels,
            "delta_f_ghz": ("freq_idx", DELTA_F),
        },
    )
    d_zfs_da = xr.DataArray(
        np.full((n_pol, NY, NX), D_ZFS),
        dims=("polarity", "y", "x"),
        coords={"polarity": pol_labels},
    )
    fold_residual_da = xr.DataArray(
        np.zeros((n_pol, NY, NX)),
        dims=("polarity", "y", "x"),
        coords={"polarity": pol_labels},
    )
    return FoldedODMR(
        folded_spectrum=folded_da,
        antisymmetric_spectrum=anti_da,
        d_zfs_map=d_zfs_da,
        fold_residual=fold_residual_da,
        settings=FoldingSettings(),
    )


def _make_mock_gpufit_result(n_pol: int, n_pixel: int, center: float) -> list:
    """Create a gpufit return value with a known centre frequency.

    Args:
        n_pol: Number of polarities.
        n_pixel: Number of spatial pixels.
        center: Centre frequency in absolute GHz (D_ZFS + delta_f).
    """
    # ESRSINGLE params: [center, width, contrast, offset]
    params = np.zeros((n_pol * n_pixel, 4), dtype=np.float32)
    params[:, 0] = center  # absolute GHz
    params[:, 1] = 0.005  # width
    params[:, 2] = 0.1  # contrast
    params[:, 3] = 1.0  # offset
    states = np.zeros(n_pol * n_pixel, dtype=np.int32)
    chi2 = np.ones(n_pol * n_pixel, dtype=np.float32) * 0.01
    iters = np.ones(n_pol * n_pixel, dtype=np.int32) * 10
    exec_time = 0.1
    return [params, states, chi2, iters, exec_time]


class _RecordingFixedBackend:
    """FitBackend that records call inputs and returns a fixed ESRSINGLE result.

    ``centers`` may be a single float (same centre for every call) or a list
    of floats consumed one per call, mirroring Mock's ``side_effect`` — used
    by tests that fit two frequency branches with different known centres.
    """

    name = "fixed"

    def __init__(self, centers: float | list[float]) -> None:
        self._centers = centers if isinstance(centers, list) else None
        self._single_center = centers if not isinstance(centers, list) else None
        self._call_index = 0
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def supports(self, model) -> bool:
        return True

    def fit(
        self,
        data,
        freq_ghz,
        initial_parameters,
        constraints,
        constraint_types,
        model,
        options,
    ) -> BackendFitOutput:
        self.calls.append(
            {
                "freq_ghz": np.asarray(freq_ghz),
                "constraints": np.asarray(constraints),
                "constraint_types": np.asarray(constraint_types),
            }
        )
        center = (
            self._centers[self._call_index] if self._centers is not None else self._single_center
        )
        self._call_index += 1

        n_freqs = data.shape[-1]
        n_fits = data.reshape((-1, n_freqs)).shape[0]
        params, states, chi2, iterations, exec_time = _make_mock_gpufit_result(
            1, n_fits, center=center
        )
        return BackendFitOutput(
            parameters=params,
            states=states,
            chi2=chi2,
            iterations=iterations,
            execution_time=exec_time,
        )


# ── Folded FitResult B111 tests (QEP-059: absolute-GHz domain) ──────────────


class TestFoldedFitResultB111:
    """Test that folded FitResult computes B111 via standard (center - D_ZFS) path."""

    def _make_result(self, center_abs_ghz: float, n_pol: int = 2) -> FitResult:
        """Build a minimal FitResult with absolute-GHz centres (as folded fits produce)."""
        n_pixel = NY * NX
        # shape: (n_pol, 1, n_pixel) -- one freq_range
        center_arr = np.full((n_pol, 1, n_pixel), center_abs_ghz, dtype=np.float64)
        chi2 = np.ones((n_pol, 1, n_pixel), dtype=np.float64) * 0.01
        return FitResult(
            parameters={"center": center_arr, "chi2": chi2},
            scan_dimensions=(NY, NX),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
            metadata={"folded_fit": True},
        )

    def test_b111_uses_dzfs_subtraction(self) -> None:
        """Centre is absolute GHz; B = (center - D_ZFS) / gamma * 1e6."""
        delta_f_ghz = 0.028  # Zeeman shift
        center_abs = D_ZFS + delta_f_ghz
        result = self._make_result(center_abs)

        expected_shift = delta_f_ghz / GAMMA_NV * 1e6  # uT

        delta = result.delta_resonance  # (polarity, y, x)
        # neg polarity gets sign=-1, pos gets +1
        assert_array_almost_equal(
            delta.sel(polarity="neg").values,
            -expected_shift * np.ones((NY, NX)),
            decimal=3,
        )
        assert_array_almost_equal(
            delta.sel(polarity="pos").values,
            +expected_shift * np.ones((NY, NX)),
            decimal=3,
        )

    def test_b111_known_value(self) -> None:
        """Symmetric centres -> remanent=0, induced=-expected."""
        delta_f_ghz = 0.028
        center_abs = D_ZFS + delta_f_ghz
        result = self._make_result(center_abs)

        expected_shift = delta_f_ghz / GAMMA_NV * 1e6
        assert_array_almost_equal(
            result.b111_remanent,
            np.zeros((NY, NX)),
            decimal=3,
        )
        assert_array_almost_equal(
            result.b111_induced,
            np.full((NY, NX), -expected_shift),
            decimal=3,
        )

    def test_b111_remanent_zero_for_symmetric_centers(self) -> None:
        """Both polarities with same centre -> remanent = 0."""
        center_abs = D_ZFS + 0.015
        result = self._make_result(center_abs)
        assert_array_almost_equal(result.b111_remanent, np.zeros((NY, NX)), decimal=6)

    def test_is_fit_result(self) -> None:
        """Folded fits return a standard FitResult (no subclass)."""
        result = self._make_result(D_ZFS + 0.02)
        assert isinstance(result, FitResult)
        assert type(result) is FitResult

    def test_model_name(self) -> None:
        """Model name remains the base model for folded fits."""
        result = self._make_result(D_ZFS + 0.02)
        assert result.model_name == "ESRSINGLE"

    def test_folded_metadata_flag(self) -> None:
        """Folded fit has metadata['folded_fit'] = True."""
        result = self._make_result(D_ZFS + 0.02)
        assert result.metadata.get("folded_fit") is True


# ── fit_folded() integration tests (mocked GPU) ─────────────────────────────


class TestFitFolded:
    """Test FitManager.fit_folded() with mocked pyGpufit."""

    def test_returns_fit_result(self) -> None:
        """fit_folded() returns a standard FitResult (not a subclass)."""
        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + 0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert isinstance(result, FitResult)
        assert type(result) is FitResult

    def test_output_shape(self) -> None:
        """b111_remanent has shape matching scan_dimensions."""
        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + 0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert result.scan_dimensions == (NY, NX)
        assert result.b111_remanent.shape == (NY, NX)
        assert result.b111_induced.shape == (NY, NX)

    def test_model_name_unchanged(self) -> None:
        """Result model_name remains the base model name."""
        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + 0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert result.model_name == "ESRSINGLE"

    def test_metadata_folded_flag(self) -> None:
        """Result metadata contains folded_fit: True."""
        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + 0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert result.metadata.get("folded_fit") is True

    def test_no_gpu_raises_dependency_error(self) -> None:
        """fit_folded() raises DependencyError when the backend is not available."""
        folded = _make_folded_odmr()
        unavailable_backend = with_forced_availability(FakeFitBackend(), available=False)
        mgr = FitManager(
            model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=unavailable_backend
        )

        with pytest.raises(DependencyError):
            mgr.fit_folded(folded)

    def test_constraints_in_absolute_ghz_domain(self) -> None:
        """The inner FitManager uses absolute-GHz centre constraints."""
        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + 0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend)
        mgr.fit_folded(folded, pixel_spacing=4e-6)

        # Inspect the constraints array passed to the backend
        constraints_arr = backend.calls[0]["constraints"]
        # column 0 = center_min, column 1 = center_max
        # With absolute_ghz mode: center_min=2.70, center_max=3.04
        center_min = float(constraints_arr[0, 0])
        center_max = float(constraints_arr[0, 1])
        assert center_min == pytest.approx(2.70, abs=0.01)
        assert center_max == pytest.approx(3.04, abs=0.01)

    def test_frequency_axis_shifted_to_absolute(self) -> None:
        """Verify that the freq axis is in absolute GHz, not delta_f."""
        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + 0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend)
        mgr.fit_folded(folded, pixel_spacing=4e-6)

        user_info = backend.calls[0]["freq_ghz"]
        # Frequencies should be D_ZFS + delta_f, so all > D_ZFS
        assert float(user_info.min()) > D_ZFS - 0.001
        assert float(user_info.min()) == pytest.approx(D_ZFS + DELTA_F[0], abs=1e-6)

    def test_preserves_explicit_center_constraints(self) -> None:
        """fit_folded() keeps explicit center bounds from the parent FitManager."""
        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + 0.028)

        mgr = FitManager(
            model_name="ESRSINGLE",
            constraints={
                "center": {
                    "vmin": D_ZFS + 0.012,
                    "vmax": D_ZFS + 0.018,
                    "constraint_type": "LOWER_UPPER",
                }
            },
            settings=MOCK_SETTINGS,
            backend=backend,
        )
        mgr.fit_folded(folded, pixel_spacing=4e-6)

        constraints_arr = backend.calls[0]["constraints"]
        center_min = float(constraints_arr[0, 0])
        center_max = float(constraints_arr[0, 1])
        assert center_min == pytest.approx(D_ZFS + 0.012, abs=1e-6)
        assert center_max == pytest.approx(D_ZFS + 0.018, abs=1e-6)


# ── Constraint conversion tests (QEP-059) ───────────────────────────────────


class TestConstraintConversion:
    """Test mT -> absolute-GHz constraint conversion."""

    def test_mt_defaults_produce_symmetric_bounds(self) -> None:
        """Default mT settings produce center bounds symmetric about D_ZFS."""
        settings = ModelConstraintsSettings()  # mt mode by default
        assert settings.constraint_units == "mt"

        delta_max_ghz = settings.center_max_mt * 1e-3 * GAMMA_NV
        expected_min = D_ZFS - delta_max_ghz
        expected_max = D_ZFS + delta_max_ghz

        from qdmpy.fitting.constraints import _mt_to_absolute_ghz

        resolved = _mt_to_absolute_ghz(settings)
        assert resolved.center_min == pytest.approx(expected_min, abs=1e-6)
        assert resolved.center_max == pytest.approx(expected_max, abs=1e-6)

    def test_mt_width_conversion(self) -> None:
        """Width mT values convert to GHz correctly."""
        settings = ModelConstraintsSettings(width_min_mt=0.01, width_max_mt=1.0)

        from qdmpy.fitting.constraints import _mt_to_absolute_ghz

        resolved = _mt_to_absolute_ghz(settings)
        expected_min = 0.01 * 1e-3 * GAMMA_NV
        expected_max = 1.0 * 1e-3 * GAMMA_NV
        assert resolved.width_min == pytest.approx(expected_min, abs=1e-8)
        assert resolved.width_max == pytest.approx(expected_max, abs=1e-8)

    def test_absolute_ghz_mode_passthrough(self) -> None:
        """absolute_ghz mode passes center/width bounds unchanged."""
        settings = ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.5,
            center_max=3.2,
            width_min=0.0002,
            width_max=0.01,
        )

        from qdmpy.fitting.constraints import ConstraintManager
        from qdmpy.fitting.models import ModelRegistry

        model = ModelRegistry.get("ESRSINGLE")
        cm = ConstraintManager(model, settings)
        constraints = cm.get_constraints()
        assert constraints["center"].vmin == pytest.approx(2.5)
        assert constraints["center"].vmax == pytest.approx(3.2)
        assert constraints["width"].vmin == pytest.approx(0.0002)
        assert constraints["width"].vmax == pytest.approx(0.01)

    def test_mt_mode_applied_in_constraint_manager(self) -> None:
        """Verify mT mode constraints are converted to absolute GHz in ConstraintManager."""
        settings = ModelConstraintsSettings(
            constraint_units="mt",
            center_max_mt=6.0,
            center_min_mt=0.0,
        )

        from qdmpy.fitting.constraints import ConstraintManager
        from qdmpy.fitting.models import ModelRegistry

        model = ModelRegistry.get("ESRSINGLE")
        cm = ConstraintManager(model, settings)
        constraints = cm.get_constraints()

        delta_max_ghz = 6.0 * 1e-3 * GAMMA_NV
        expected_min = D_ZFS - delta_max_ghz
        expected_max = D_ZFS + delta_max_ghz
        assert constraints["center"].vmin == pytest.approx(expected_min, abs=1e-6)
        assert constraints["center"].vmax == pytest.approx(expected_max, abs=1e-6)

    def test_mt_center_window_applied_per_branch(self) -> None:
        """center_min_mt enforces a true [min, max] mT window per frequency branch."""
        settings = QDMpySettings(
            fit=FitSettings(estimator="LSE", max_number_iterations=100, tolerance=1e-6),
            model=ModelSettings(
                constraints=ModelConstraintsSettings(
                    constraint_units="mt",
                    center_min_mt=2.0,
                    center_max_mt=7.0,
                    center_type="LOWER_UPPER",
                )
            ),
        )

        n_pol, ny, nx, n_freq = 2, 2, 2, 16
        low_freq = np.linspace(D_ZFS - 0.08, D_ZFS - 0.005, n_freq)
        high_freq = np.linspace(D_ZFS + 0.005, D_ZFS + 0.08, n_freq)
        freqs = np.stack([low_freq, high_freq], axis=0)

        data = xr.DataArray(
            np.ones((n_pol, 2, ny, nx, n_freq), dtype=np.float32),
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={"polarity": ["neg", "pos"], "freq_range": ["low", "high"]},
        )

        low_center = D_ZFS - (3.0 * 1e-3 * GAMMA_NV)
        high_center = D_ZFS + (3.0 * 1e-3 * GAMMA_NV)
        backend = _RecordingFixedBackend([low_center, high_center])

        mgr = FitManager(model_name="ESRSINGLE", settings=settings, backend=backend)
        _ = mgr.fit(data, freqs)

        delta_min = 2.0 * 1e-3 * GAMMA_NV
        delta_max = 7.0 * 1e-3 * GAMMA_NV

        low_constraints = backend.calls[0]["constraints"]
        high_constraints = backend.calls[1]["constraints"]

        assert float(low_constraints[0, 0]) == pytest.approx(D_ZFS - delta_max, abs=1e-6)
        assert float(low_constraints[0, 1]) == pytest.approx(D_ZFS - delta_min, abs=1e-6)

        assert float(high_constraints[0, 0]) == pytest.approx(D_ZFS + delta_min, abs=1e-6)
        assert float(high_constraints[0, 1]) == pytest.approx(D_ZFS + delta_max, abs=1e-6)

    def test_mt_center_window_applied_in_folded_fit(self) -> None:
        """Folded fit uses the high-branch window when center_min_mt > 0."""
        settings = QDMpySettings(
            fit=FitSettings(estimator="LSE", max_number_iterations=100, tolerance=1e-6),
            model=ModelSettings(
                constraints=ModelConstraintsSettings(
                    constraint_units="mt",
                    center_min_mt=2.0,
                    center_max_mt=7.0,
                    center_type="LOWER_UPPER",
                )
            ),
        )

        folded = _make_folded_odmr()
        backend = _RecordingFixedBackend(D_ZFS + (3.0 * 1e-3 * GAMMA_NV))

        mgr = FitManager(model_name="ESRSINGLE", settings=settings, backend=backend)
        _ = mgr.fit_folded(folded, pixel_spacing=4e-6)

        delta_min = 2.0 * 1e-3 * GAMMA_NV
        delta_max = 7.0 * 1e-3 * GAMMA_NV
        constraints = backend.calls[0]["constraints"]

        assert float(constraints[0, 0]) == pytest.approx(D_ZFS + delta_min, abs=1e-6)
        assert float(constraints[0, 1]) == pytest.approx(D_ZFS + delta_max, abs=1e-6)
