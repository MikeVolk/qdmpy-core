"""Characterization/parity tests for the unified fit pipeline (QEP-070).

These tests pin FitManager.fit()/fit_folded()'s observable behavior — output
shapes, quality metrics, and the exact arrays handed to the FitBackend — so
the QEP-070 decomposition (god-method extraction, non-mutating constraints,
folded-path unification) can be verified to preserve behavior at every phase.

Tests use injectable FitBackend fakes (QEP-068); no GPU. `guess_model` is
patched only where auto-model detection input needs to be inspected.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from qdmpy.constants import D_ZFS, GAMMA_NV
from qdmpy.exceptions import DataValidationError
from qdmpy.fitting.backends import BackendFitOutput
from qdmpy.fitting.manager import FitManager
from qdmpy.fitting.models import ESRSINGLE
from qdmpy.fitting.result import FitResult
from qdmpy.odmr.folding import FoldedODMR, FoldingSettings
from qdmpy.settings import (
    FitSettings,
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
)
from qdmpy.testing import FakeFitBackend

# ── Settings ─────────────────────────────────────────────────────────────────

MOCK_SETTINGS = QDMpySettings(
    fit=FitSettings(estimator="LSE", max_number_iterations=100, tolerance=1e-6),
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

MT_SETTINGS = QDMpySettings(
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

NY, NX, N_FREQ = 2, 2, 16


class _RecordingEchoBackend:
    """FitBackend that records call inputs and echoes back initial_parameters.

    Same echo behavior as FakeFitBackend, but retains the constraints/
    constraint_types/freq_ghz arrays received on every fit() call so tests can
    assert on exactly what the pipeline handed to the backend.
    """

    name = "recording-echo"

    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

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
                "freq_ghz": np.array(freq_ghz),
                "constraints": np.array(constraints),
                "constraint_types": np.array(constraint_types),
            }
        )
        n_freqs = data.shape[-1]
        n_fits = data.reshape((-1, n_freqs)).shape[0]
        n_params = initial_parameters.shape[-1]
        params = np.asarray(initial_parameters, dtype=np.float32).reshape((n_fits, n_params))
        return BackendFitOutput(
            parameters=params,
            states=np.zeros(n_fits, dtype=np.int32),
            chi2=np.zeros(n_fits, dtype=np.float32),
            iterations=np.ones(n_fits, dtype=np.int32),
            execution_time=0.0,
        )


def _make_two_frange_data() -> tuple[xr.DataArray, np.ndarray]:
    """Build (n_pol=2, n_frange=2, 2, 2, N_FREQ) data straddling D_ZFS."""
    low_freq = np.linspace(D_ZFS - 0.08, D_ZFS - 0.005, N_FREQ)
    high_freq = np.linspace(D_ZFS + 0.005, D_ZFS + 0.08, N_FREQ)
    freqs = np.stack([low_freq, high_freq], axis=0)
    data = xr.DataArray(
        np.ones((2, 2, NY, NX, N_FREQ), dtype=np.float32),
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={"polarity": ["neg", "pos"], "freq_range": ["low", "high"]},
    )
    return data, freqs


def _make_folded_odmr(n_pol: int = 2, n_df: int = 20) -> FoldedODMR:
    """Build a minimal synthetic FoldedODMR object."""
    pol_labels = ["neg", "pos"][:n_pol]
    delta_f = np.linspace(0.005, 0.060, n_df)
    spec = np.ones((n_pol, NY, NX, n_df), dtype=np.float32)

    folded_da = xr.DataArray(
        spec,
        dims=("polarity", "y", "x", "freq_idx"),
        coords={"polarity": pol_labels, "delta_f_ghz": ("freq_idx", delta_f)},
    )
    anti_da = xr.DataArray(
        np.zeros_like(spec),
        dims=("polarity", "y", "x", "freq_idx"),
        coords={"polarity": pol_labels, "delta_f_ghz": ("freq_idx", delta_f)},
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


class TestFitCharacterization:
    """Pin fit()'s output shape/keys/metrics contract."""

    def test_parameters_keys_and_shapes(self) -> None:
        data, freqs = _make_two_frange_data()
        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        result = mgr.fit(data, freqs)

        assert isinstance(result, FitResult)
        expected_keys = {"center", "width", "contrast", "offset", "chi2", "states"}
        assert set(result.parameters.keys()) == expected_keys
        for key, arr in result.parameters.items():
            assert arr.shape == (2, 2, NY, NX), f"{key} has shape {arr.shape}"
        assert result.scan_dimensions == (NY, NX)

    def test_quality_metrics_contract(self) -> None:
        data, freqs = _make_two_frange_data()
        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        result = mgr.fit(data, freqs)

        metrics = result.metadata["quality_metrics"]
        expected_keys = {
            "mean_chi2",
            "median_chi2",
            "std_chi2",
            "convergence_rate",
            "n_pixels",
            "n_converged",
            "total_fit_time",
        }
        assert set(metrics.keys()) == expected_keys
        assert metrics["mean_chi2"] == pytest.approx(0.0)
        assert metrics["convergence_rate"] == pytest.approx(1.0)
        assert metrics["n_pixels"] == 2 * 2 * NY * NX
        assert metrics["n_converged"] == 2 * 2 * NY * NX
        assert metrics["total_fit_time"] == pytest.approx(0.0)
        assert "fit_timestamp" in result.metadata


class TestFoldedFitCharacterization:
    """Pin fit_folded()'s output shape/metadata contract."""

    def test_parameters_keys_and_shapes(self) -> None:
        folded = _make_folded_odmr()
        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert isinstance(result, FitResult)
        expected_keys = {"center", "width", "contrast", "offset", "chi2", "states"}
        assert set(result.parameters.keys()) == expected_keys
        for key, arr in result.parameters.items():
            assert arr.shape == (2, 1, NY, NX), f"{key} has shape {arr.shape}"
        assert result.metadata["folded_fit"] is True
        assert result.scan_dimensions == (NY, NX)


class TestBackendInputIdentityAcrossRepeatFits:
    """The pipeline must not leak state between independent fit() calls."""

    def test_repeat_fit_receives_identical_inputs(self) -> None:
        data, freqs = _make_two_frange_data()
        backend = _RecordingEchoBackend()
        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend)

        mgr.fit(data, freqs)
        first_calls = backend.calls
        backend.calls = []
        mgr.fit(data, freqs)
        second_calls = backend.calls

        assert len(first_calls) == len(second_calls) == 2
        for call_a, call_b in zip(first_calls, second_calls, strict=True):
            np.testing.assert_array_equal(call_a["freq_ghz"], call_b["freq_ghz"])
            np.testing.assert_array_equal(call_a["constraints"], call_b["constraints"])
            np.testing.assert_array_equal(call_a["constraint_types"], call_b["constraint_types"])


class TestPerFrangeMtCenterWindow:
    """Pin the per-branch mT center-window bounds handed to the backend."""

    def test_low_and_high_branch_windows(self) -> None:
        data, freqs = _make_two_frange_data()
        backend = _RecordingEchoBackend()
        mgr = FitManager(model_name="ESRSINGLE", settings=MT_SETTINGS, backend=backend)
        mgr.fit(data, freqs)

        delta_min = 2.0 * 1e-3 * GAMMA_NV
        delta_max = 7.0 * 1e-3 * GAMMA_NV

        low_constraints = backend.calls[0]["constraints"]
        high_constraints = backend.calls[1]["constraints"]

        assert float(low_constraints[0, 0]) == pytest.approx(D_ZFS - delta_max, abs=1e-6)
        assert float(low_constraints[0, 1]) == pytest.approx(D_ZFS - delta_min, abs=1e-6)
        assert float(high_constraints[0, 0]) == pytest.approx(D_ZFS + delta_min, abs=1e-6)
        assert float(high_constraints[0, 1]) == pytest.approx(D_ZFS + delta_max, abs=1e-6)

    def test_shared_constraint_manager_not_mutated_after_fit(self) -> None:
        """fit() must not leak a per-range mT window into the shared ConstraintManager."""
        data, freqs = _make_two_frange_data()
        backend = _RecordingEchoBackend()
        mgr = FitManager(model_name="ESRSINGLE", settings=MT_SETTINGS, backend=backend)

        before = mgr.constraints["center"]
        mgr.fit(data, freqs)
        after = mgr.constraints["center"]

        assert after == before

        # A second fit must reproduce the exact same per-branch windows, proving
        # the manager's stored constraints were never overwritten by the first call.
        backend.calls = []
        mgr.fit(data, freqs)
        delta_min = 2.0 * 1e-3 * GAMMA_NV
        delta_max = 7.0 * 1e-3 * GAMMA_NV
        assert float(backend.calls[0]["constraints"][0, 0]) == pytest.approx(
            D_ZFS - delta_max, abs=1e-6
        )
        assert float(backend.calls[1]["constraints"][0, 0]) == pytest.approx(
            D_ZFS + delta_min, abs=1e-6
        )


class TestFoldedNonFoldedParity:
    """fit_folded() must reach the backend with the same effective inputs as a
    manually-equivalent fit() call (QEP-070 phase 4: no second FitManager).
    """

    def test_backend_receives_equivalent_inputs(self) -> None:
        folded = _make_folded_odmr()

        backend_folded = _RecordingEchoBackend()
        mgr_folded = FitManager(
            model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend_folded
        )
        result_folded = mgr_folded.fit_folded(folded, pixel_spacing=4e-6)

        backend_direct = _RecordingEchoBackend()
        mgr_direct = FitManager(
            model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=backend_direct
        )
        mgr_direct.set_constraints("contrast", vmin=0.001, vmax=1.0, constraint_type="LOWER_UPPER")
        mgr_direct.set_constraints("offset", vmin=-0.5, vmax=3.0, constraint_type="LOWER_UPPER")
        data_xr, freq_2d = folded.to_fit_inputs()
        result_direct = mgr_direct.fit(data_xr, freq_2d, pixel_spacing=4e-6)

        np.testing.assert_array_equal(
            backend_folded.calls[0]["freq_ghz"], backend_direct.calls[0]["freq_ghz"]
        )
        np.testing.assert_array_equal(
            backend_folded.calls[0]["constraints"], backend_direct.calls[0]["constraints"]
        )
        np.testing.assert_array_equal(
            backend_folded.calls[0]["constraint_types"], backend_direct.calls[0]["constraint_types"]
        )
        for key in result_direct.parameters:
            np.testing.assert_allclose(
                result_folded.parameters[key], result_direct.parameters[key]
            )
        assert result_folded.metadata["folded_fit"] is True
        assert "folded_fit" not in result_direct.metadata

    def test_no_second_fit_manager_constructed(self) -> None:
        folded = _make_folded_odmr()
        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())

        with patch.object(FitManager, "__init__", wraps=FitManager.__init__) as mock_init:
            mgr.fit_folded(folded, pixel_spacing=4e-6)

        mock_init.assert_not_called()


class TestFoldedAutoModelDetection:
    """fit_folded()'s raw_data controls which array is handed to guess_model()."""

    def test_uses_raw_data_when_given(self) -> None:
        folded = _make_folded_odmr(n_df=20)
        raw_data = np.ones((2, 2, NY, NX, 30), dtype=np.float32)
        mgr = FitManager(model_name="auto", settings=MOCK_SETTINGS, backend=FakeFitBackend())

        with patch(
            "qdmpy.fitting.manager.guess_model", return_value=ESRSINGLE()
        ) as mock_guess:
            mgr.fit_folded(folded, pixel_spacing=4e-6, raw_data=raw_data)

        detection_arg = mock_guess.call_args[0][0]
        assert detection_arg.shape[-1] == 30

    def test_falls_back_to_folded_data_without_raw_data(self) -> None:
        folded = _make_folded_odmr(n_df=20)
        mgr = FitManager(model_name="auto", settings=MOCK_SETTINGS, backend=FakeFitBackend())

        with patch(
            "qdmpy.fitting.manager.guess_model", return_value=ESRSINGLE()
        ) as mock_guess:
            mgr.fit_folded(folded, pixel_spacing=4e-6)

        detection_arg = mock_guess.call_args[0][0]
        assert detection_arg.shape[-1] == 20


class TestFoldedFitErrorContractParity:
    """fit_folded() must validate inputs with the same errors as fit()."""

    def test_too_few_frequency_points_raises(self) -> None:
        folded = _make_folded_odmr(n_df=5)
        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        with pytest.raises(DataValidationError, match="at least"):
            mgr.fit_folded(folded, pixel_spacing=4e-6)
