"""Tests for QDMpy.result.QDMResult container."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from qdmpy.fitting.result import FitResult
from qdmpy.result import QDMResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_pol_fit_result() -> FitResult:
    """FitResult with two polarities — supports b111 computation.

    center is (n_pol, n_frange, n_pixels) so delta_resonance / b111 work.
    chi2 is flat (n_pixels,) so get_parameter_map('chi2') works.
    """
    n_pol, n_frange, n_pixels = 2, 2, 25
    rng = np.random.default_rng(0)
    center = rng.uniform(2.82, 2.92, (n_pol, n_frange, n_pixels))
    return FitResult(
        parameters={
            "center": center,
            "chi2": rng.random(n_pixels),
        },
        scan_dimensions=(5, 5),
        pixel_spacing=4e-6,
        model_name="ESR14N",
    )


@pytest.fixture
def qdm_result(two_pol_fit_result: FitResult) -> QDMResult:
    return QDMResult(fit_result=two_pol_fit_result)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestQDMResultConstruction:
    def test_wraps_fit_result(self, two_pol_fit_result: FitResult) -> None:
        result = QDMResult(fit_result=two_pol_fit_result)
        assert result.fit_result is two_pol_fit_result

    def test_nv_axis_default_is_none(self, two_pol_fit_result: FitResult) -> None:
        result = QDMResult(fit_result=two_pol_fit_result)
        assert result.nv_axis is None

    def test_nv_axis_can_be_set(self, two_pol_fit_result: FitResult) -> None:
        axis = (0.0, 0.8164966, 0.5773503)
        result = QDMResult(fit_result=two_pol_fit_result, nv_axis=axis)
        assert result.nv_axis == axis

    def test_rejects_non_fitresult(self) -> None:
        with pytest.raises(ValidationError):
            QDMResult(fit_result="garbage")

    def test_repr(self, qdm_result: QDMResult) -> None:
        r = repr(qdm_result)
        assert "QDMResult" in r
        assert "ESR14N" in r


# ---------------------------------------------------------------------------
# Delegation to FitResult
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_scan_dimensions(self, qdm_result: QDMResult) -> None:
        assert qdm_result.scan_dimensions == (5, 5)

    def test_pixel_spacing(self, qdm_result: QDMResult) -> None:
        assert qdm_result.pixel_spacing == 4e-6

    def test_model_name(self, qdm_result: QDMResult) -> None:
        assert qdm_result.model_name == "ESR14N"

    def test_chi2_delegates(self, qdm_result: QDMResult) -> None:
        np.testing.assert_array_equal(qdm_result.chi2, qdm_result.fit_result.chi2)

    def test_centers_delegates(self, qdm_result: QDMResult) -> None:
        np.testing.assert_array_equal(qdm_result.centers, qdm_result.fit_result.centers)

    def test_b111_remanent_is_ndarray(self, qdm_result: QDMResult) -> None:
        arr = qdm_result.b111_remanent
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 5)

    def test_b111_induced_is_ndarray(self, qdm_result: QDMResult) -> None:
        arr = qdm_result.b111_induced
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 5)

    def test_b111_is_dataset(self, qdm_result: QDMResult) -> None:
        ds = qdm_result.b111
        assert isinstance(ds, xr.Dataset)
        assert "remanent" in ds
        assert "induced" in ds

    def test_get_parameter_map(self, qdm_result: QDMResult) -> None:
        # chi2 is 1D (n_pixels,) so reshape to scan_dimensions works
        m = qdm_result.get_parameter_map("chi2")
        assert m.shape == (5, 5)

    def test_get_fit_quality_metrics(self, qdm_result: QDMResult) -> None:
        metrics = qdm_result.get_fit_quality_metrics()
        assert "mean_chi2" in metrics
        assert "n_pixels" in metrics


# ---------------------------------------------------------------------------
# Lazy MagneticMap
# ---------------------------------------------------------------------------


class TestMagneticMap:
    def test_magnetic_map_not_computed_on_init(self, qdm_result: QDMResult) -> None:
        assert qdm_result._magnetic_map_cache is None

    def test_magnetic_map_computed_on_access(self, qdm_result: QDMResult) -> None:
        mag_map = qdm_result.magnetic_map
        assert mag_map is not None
        assert qdm_result._magnetic_map_cache is not None

    def test_magnetic_map_cached(self, qdm_result: QDMResult) -> None:
        first = qdm_result.magnetic_map
        second = qdm_result.magnetic_map
        assert first is second

    def test_magnetic_map_has_bz(self, qdm_result: QDMResult) -> None:
        bz = qdm_result.magnetic_map.bz
        assert isinstance(bz, xr.DataArray)
        assert bz.shape == (5, 5)

    def test_magnetic_map_uses_settings_nv_axis_by_default(self, qdm_result: QDMResult) -> None:
        """When nv_axis is None, from_b111 uses settings.nv.axis."""
        with patch("qdmpy.magnetic_map.MagneticMap.from_b111") as mock_from_b111:
            mock_from_b111.return_value = MagicMock()
            _ = qdm_result.magnetic_map
            call_kwargs = mock_from_b111.call_args
            # nv_axis=None is passed; MagneticMap.from_b111 resolves it from settings
            assert call_kwargs[1].get("nv_axis") is None

    def test_magnetic_map_passes_custom_nv_axis(self, two_pol_fit_result: FitResult) -> None:
        axis = (0.1, 0.8, 0.6)
        result = QDMResult(fit_result=two_pol_fit_result, nv_axis=axis)
        with patch("qdmpy.magnetic_map.MagneticMap.from_b111") as mock_from_b111:
            mock_from_b111.return_value = MagicMock()
            _ = result.magnetic_map
            call_kwargs = mock_from_b111.call_args
            assert call_kwargs[1].get("nv_axis") == axis


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """NPZ persistence via qdmpy.io.save_npz / load_npz (QEP-008)."""

    def test_save_creates_npz(self, qdm_result: QDMResult, tmp_path: Path) -> None:
        from qdmpy.io import save_npz

        out = tmp_path / "result.npz"
        save_npz(qdm_result, out)
        assert out.exists()

    def test_roundtrip_no_nv_axis(self, qdm_result: QDMResult, tmp_path: Path) -> None:
        from qdmpy.io import load_npz, save_npz

        out = tmp_path / "result.npz"
        save_npz(qdm_result, out)
        loaded = load_npz(out)
        assert loaded.model_name == qdm_result.model_name
        assert loaded.pixel_spacing == qdm_result.pixel_spacing
        assert loaded.scan_dimensions == qdm_result.scan_dimensions
        assert loaded.nv_axis is None

    def test_roundtrip_with_nv_axis(self, two_pol_fit_result: FitResult, tmp_path: Path) -> None:
        from qdmpy.io import load_npz, save_npz

        axis = (0.0, 0.8164966, 0.5773503)
        original = QDMResult(fit_result=two_pol_fit_result, nv_axis=axis)
        out = tmp_path / "result.npz"
        save_npz(original, out)
        loaded = load_npz(out)
        assert loaded.nv_axis is not None
        assert len(loaded.nv_axis) == 3
        np.testing.assert_allclose(loaded.nv_axis, axis)

    def test_roundtrip_b111_remanent(self, qdm_result: QDMResult, tmp_path: Path) -> None:
        from qdmpy.io import load_npz, save_npz

        out = tmp_path / "result.npz"
        save_npz(qdm_result, out)
        loaded = load_npz(out)
        np.testing.assert_allclose(loaded.b111_remanent, qdm_result.b111_remanent)

    def test_save_is_pickle_free(self, qdm_result: QDMResult, tmp_path: Path) -> None:
        """save_npz() produces a file loadable without pickle."""
        from qdmpy.io import save_npz

        out = tmp_path / "result.npz"
        save_npz(qdm_result, out)
        data = np.load(out, allow_pickle=False)
        assert "__meta__" in data.files
