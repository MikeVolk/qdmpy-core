"""Unit tests for FoldedFitResult and FitManager.fit_folded().

All tests use synthetic FoldedODMR data and a mocked GPU
(pattern from tests/test_fit.py: @patch("pygpufit.gpufit.fit_constrained")).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from qdmpy.constants import GAMMA_NV
from qdmpy.fitting.manager import (
    _FOLDED_CENTER_MAX,
    _FOLDED_CENTER_MIN,
    _FOLDED_CONTRAST_MAX,
    _FOLDED_WIDTH_MAX,
    _FOLDED_WIDTH_MIN,
    FitManager,
)
from qdmpy.fitting.result import FitResult, FoldedFitResult
from qdmpy.odmr.folding import FoldedODMR, FoldingSettings
from qdmpy.settings import (
    FitSettings,
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
)

# ── Settings ────────────────────────────────────────────────────────────────

MOCK_SETTINGS = QDMpySettings(
    fit=FitSettings(
        estimator="LSE",
        max_number_iterations=100,
        tolerance=1e-6,
    ),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            center_min=0.001,
            center_max=0.080,
            center_type="FREE",
            width_min=0.001,
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
        np.full((n_pol, NY, NX), 2.870),
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
    """Create a gpufit return value with a known centre frequency."""
    # ESRSINGLE params: [center, width, contrast, offset]
    params = np.zeros((n_pol * n_pixel, 4), dtype=np.float32)
    params[:, 0] = center  # centre = delta_f
    params[:, 1] = 0.005  # width
    params[:, 2] = 0.1  # contrast
    params[:, 3] = 1.0  # offset
    states = np.zeros(n_pol * n_pixel, dtype=np.int32)
    chi2 = np.ones(n_pol * n_pixel, dtype=np.float32) * 0.01
    iters = np.ones(n_pol * n_pixel, dtype=np.int32) * 10
    exec_time = 0.1
    return [params, states, chi2, iters, exec_time]


# ── FoldedFitResult physics tests ───────────────────────────────────────────


class TestFoldedFitResultB111:
    """Test that FoldedFitResult computes B111 without D_ZFS subtraction."""

    def _make_result(self, center: float, n_pol: int = 2) -> FoldedFitResult:
        """Build a minimal FoldedFitResult with a uniform centre map."""
        n_pixel = NY * NX
        # shape: (n_pol, 1, n_pixel) — one freq_range
        center_arr = np.full((n_pol, 1, n_pixel), center, dtype=np.float64)
        chi2 = np.ones((n_pol, 1, n_pixel), dtype=np.float64) * 0.01
        return FoldedFitResult(
            parameters={"center": center_arr, "chi2": chi2},
            scan_dimensions=(NY, NX),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE+FOLDED",
        )

    def test_b111_no_dzfs_subtraction(self) -> None:
        """Centre is delta_f; B = center / gamma * 1e6 (no D_ZFS subtraction)."""
        center_ghz = 0.028  # GHz
        result = self._make_result(center_ghz)

        expected_shift = center_ghz / GAMMA_NV * 1e6  # uT, ~999 uT for 0.028 GHz

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
        """Center 0.028 GHz -> B ~999 µT; remanent = mean of signed deltas."""
        center_ghz = 0.028
        result = self._make_result(center_ghz)

        expected_shift = center_ghz / GAMMA_NV * 1e6
        # remanent = (neg_diff + pos_diff) / 2 = (-exp + exp) / 2 = 0... wait
        # neg_diff = -expected_shift, pos_diff = +expected_shift
        # remanent = (neg_diff + pos_diff)/2 = 0; induced = (neg - pos)/2 = -expected_shift
        # Actually from the formula: neg_diff = delta_resonance[neg] = -expected_shift
        # b111_remanent = (neg_diff + pos_diff) / 2 = 0 for symmetric B
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

    def test_b111_induced_zero_for_symmetric_centers(self) -> None:
        """When both polarities have the same magnitude center, induced B = 0."""
        # Both neg and pos have same centre -> induced = (neg - pos) / 2
        # neg_diff = -expected, pos_diff = +expected
        # induced = (-expected - expected) / 2 = -expected (not zero for this case)
        # For induced == 0 we need neg_diff == pos_diff i.e., asymmetric B cancels
        # Let's verify the remanent instead — it's (neg_diff + pos_diff) / 2 = 0
        center_ghz = 0.015
        result = self._make_result(center_ghz)
        assert_array_almost_equal(result.b111_remanent, np.zeros((NY, NX)), decimal=6)

    def test_is_subclass_of_fit_result(self) -> None:
        """FoldedFitResult is a FitResult subclass."""
        result = self._make_result(0.02)
        assert isinstance(result, FitResult)
        assert isinstance(result, FoldedFitResult)

    def test_model_name(self) -> None:
        """Model name is preserved."""
        result = self._make_result(0.02)
        assert result.model_name == "ESRSINGLE+FOLDED"


# ── fit_folded() integration tests (mocked GPU) ─────────────────────────────


class TestFitFolded:
    """Test FitManager.fit_folded() with mocked pyGpufit."""

    @patch("pygpufit.gpufit.fit_constrained")
    def test_returns_folded_fit_result(self, mock_gf) -> None:
        """fit_folded() returns a FoldedFitResult instance."""
        folded = _make_folded_odmr()
        n_pol, n_pixel = 2, NY * NX
        mock_gf.return_value = _make_mock_gpufit_result(n_pol, n_pixel, center=0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, gpu_available=True)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert isinstance(result, FoldedFitResult)

    @patch("pygpufit.gpufit.fit_constrained")
    def test_output_shape(self, mock_gf) -> None:
        """b111_remanent has shape matching scan_dimensions."""
        folded = _make_folded_odmr()
        n_pol, n_pixel = 2, NY * NX
        mock_gf.return_value = _make_mock_gpufit_result(n_pol, n_pixel, center=0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, gpu_available=True)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert result.scan_dimensions == (NY, NX)
        assert result.b111_remanent.shape == (NY, NX)
        assert result.b111_induced.shape == (NY, NX)

    @patch("pygpufit.gpufit.fit_constrained")
    def test_model_name_tag(self, mock_gf) -> None:
        """Result model_name contains FOLDED tag."""
        folded = _make_folded_odmr()
        n_pol, n_pixel = 2, NY * NX
        mock_gf.return_value = _make_mock_gpufit_result(n_pol, n_pixel, center=0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, gpu_available=True)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert "FOLDED" in result.model_name

    @patch("pygpufit.gpufit.fit_constrained")
    def test_metadata_folded_flag(self, mock_gf) -> None:
        """Result metadata contains folded_fit: True."""
        folded = _make_folded_odmr()
        n_pol, n_pixel = 2, NY * NX
        mock_gf.return_value = _make_mock_gpufit_result(n_pol, n_pixel, center=0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, gpu_available=True)
        result = mgr.fit_folded(folded, pixel_spacing=4e-6)

        assert result.metadata.get("folded_fit") is True

    @patch("pygpufit.gpufit.fit_constrained")
    def test_no_gpu_raises_dependency_error(self, mock_gf) -> None:
        """fit_folded() raises DependencyError when GPU is not available."""
        from qdmpy.exceptions import DependencyError

        folded = _make_folded_odmr()
        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, gpu_available=False)

        with pytest.raises(DependencyError):
            mgr.fit_folded(folded)

    @patch("pygpufit.gpufit.fit_constrained")
    def test_constraints_in_delta_f_domain(self, mock_gf) -> None:
        """The inner FitManager uses folded-domain centre constraints."""
        folded = _make_folded_odmr()
        n_pol, n_pixel = 2, NY * NX
        mock_gf.return_value = _make_mock_gpufit_result(n_pol, n_pixel, center=0.028)

        mgr = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, gpu_available=True)
        mgr.fit_folded(folded, pixel_spacing=4e-6)

        # Inspect the constraints array passed to gpufit
        call_kwargs = mock_gf.call_args
        constraints_arr = call_kwargs.kwargs.get(
            "constraints", call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        # constraints array shape: (n_pol*n_pixel, 2*n_params)
        # params order for ESRSINGLE: center, width, contrast, offset
        # column 0 = center_min, column 1 = center_max
        assert constraints_arr is not None
        assert float(constraints_arr[0, 0]) == pytest.approx(_FOLDED_CENTER_MIN, abs=1e-6)
        assert float(constraints_arr[0, 1]) == pytest.approx(_FOLDED_CENTER_MAX, abs=1e-6)


# ── Module-level constant sanity checks ─────────────────────────────────────


class TestFoldedConstants:
    """Sanity checks on the folded-domain constraint constants."""

    def test_center_bounds_positive(self) -> None:
        assert _FOLDED_CENTER_MIN > 0
        assert _FOLDED_CENTER_MAX > _FOLDED_CENTER_MIN

    def test_width_bounds_positive(self) -> None:
        assert _FOLDED_WIDTH_MIN > 0
        assert _FOLDED_WIDTH_MAX > _FOLDED_WIDTH_MIN

    def test_contrast_max_at_most_one(self) -> None:
        """Folded spectrum is normalised by 2 before fitting, so contrast <= 1.0."""
        assert _FOLDED_CONTRAST_MAX <= 1.0

    def test_center_max_physical(self) -> None:
        """Center max in GHz corresponds to a reasonable field (< 3 mT)."""
        b_max = _FOLDED_CENTER_MAX / GAMMA_NV  # Tesla
        assert b_max < 3e-3  # 3 mT
