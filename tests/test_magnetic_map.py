"""Tests for qdmpy.magnetic_map — MagneticMap and FieldReconstructor."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from qdmpy.magnetic_map import FieldReconstructor, MagneticMap, _reconstruct_bxyz
from qdmpy.settings import NvSettings, QDMpySettings

# ---------------------------------------------------------------------------
# _reconstruct_bxyz (Fourier inversion)
# ---------------------------------------------------------------------------


class TestReconstructBxyz:
    """Tests for the low-level Fourier inversion function."""

    def test_output_shapes(self) -> None:
        b111 = np.random.default_rng(42).standard_normal((16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, pixel_spacing=4e-6, nv_axis=(0, 0, 1), epsilon=1e-30)
        assert bx.shape == (16, 16)
        assert by.shape == (16, 16)
        assert bz.shape == (16, 16)

    def test_z_aligned_nv_gives_bz_correlated_with_input(self) -> None:
        """When NV axis is (0, 0, 1), bz should strongly correlate with b111."""
        rng = np.random.default_rng(7)
        b111 = rng.standard_normal((32, 32)) * 10.0
        bx, by, bz = _reconstruct_bxyz(b111, pixel_spacing=4e-6, nv_axis=(0, 0, 1), epsilon=1e-30)
        # Correlation coefficient should be very high
        corr = np.corrcoef(bz.ravel(), b111.ravel())[0, 1]
        assert corr > 0.99

    def test_output_is_real(self) -> None:
        b111 = np.ones((8, 8))
        bx, by, bz = _reconstruct_bxyz(b111, pixel_spacing=1e-5, nv_axis=(1, 1, 1), epsilon=1e-30)
        assert bx.dtype in (np.float64, np.float32)
        assert by.dtype in (np.float64, np.float32)
        assert bz.dtype in (np.float64, np.float32)


# ---------------------------------------------------------------------------
# MagneticMap.from_b111
# ---------------------------------------------------------------------------


class TestMagneticMap:
    """Tests for MagneticMap construction and properties."""

    @pytest.fixture
    def b111_da(self) -> xr.DataArray:
        rng = np.random.default_rng(42)
        return xr.DataArray(
            rng.standard_normal((16, 16)) * 5.0,
            dims=("y", "x"),
            attrs={"pixel_spacing": 4e-6, "units": "µT"},
        )

    def test_from_b111_creates_all_components(self, b111_da) -> None:
        mm = MagneticMap.from_b111(b111_da, nv_axis=(0, 0, 1), epsilon=1e-30)
        assert mm.bx is not None
        assert mm.by is not None
        assert mm.bz is not None
        assert mm.btotal is not None
        assert mm.nv_axis == (0, 0, 1)

    def test_btotal_nonnegative(self, b111_da) -> None:
        mm = MagneticMap.from_b111(b111_da, nv_axis=(0, 0, 1), epsilon=1e-30)
        assert np.all(mm.btotal.values >= 0)

    def test_to_dataset_has_expected_vars(self, b111_da) -> None:
        mm = MagneticMap.from_b111(b111_da, nv_axis=(0, 0, 1), epsilon=1e-30)
        ds = mm.to_dataset()
        for var in ("b111", "Bx", "By", "Bz", "Btotal"):
            assert var in ds

    def test_missing_pixel_spacing_raises(self) -> None:
        da = xr.DataArray(np.ones((4, 4)), dims=("y", "x"))
        with pytest.raises(ValueError, match="pixel_spacing"):
            MagneticMap.from_b111(da, nv_axis=(0, 0, 1))

    def test_custom_reconstructor(self, b111_da) -> None:
        """A custom FieldReconstructor is used when provided."""

        class ZeroReconstructor:
            def reconstruct(self, b111, nv_axis):
                zeros = xr.DataArray(np.zeros_like(b111.values), dims=b111.dims)
                return xr.Dataset(
                    {
                        "bx": zeros,
                        "by": zeros,
                        "bz": zeros,
                        "btotal": zeros,
                    }
                )

        assert isinstance(ZeroReconstructor(), FieldReconstructor)
        mm = MagneticMap.from_b111(b111_da, nv_axis=(0, 0, 1), reconstructor=ZeroReconstructor())
        np.testing.assert_array_equal(mm.bz.values, 0)

    def test_explicit_settings_avoid_global_lookup(self, b111_da) -> None:
        settings = QDMpySettings(nv=NvSettings(axis=(0, 0, 1), epsilon=1e-30))

        with patch("qdmpy.settings.get_settings", side_effect=AssertionError("global lookup used")):
            mm = MagneticMap.from_b111(b111_da, settings=settings)

        assert mm.nv_axis == (0, 0, 1)

    def test_frozen_dataclass(self, b111_da) -> None:
        mm = MagneticMap.from_b111(b111_da, nv_axis=(0, 0, 1), epsilon=1e-30)
        with pytest.raises(AttributeError):
            mm.nv_axis = (1, 0, 0)  # type: ignore[misc]

    def test_display_accepts_documented_imshow_kwargs(self, b111_da) -> None:
        """Regression test: display()'s documented **imshow_kwargs passthrough
        used to raise TypeError -- it forwarded to plot_magnetic_component(),
        whose signature had no **kwargs at all.
        """
        import matplotlib.pyplot as plt

        original_show = plt.show
        plt.show = lambda: None
        try:
            mm = MagneticMap.from_b111(b111_da, nv_axis=(0, 0, 1), epsilon=1e-30)
            mm.display("Bz", cmap="plasma")
        finally:
            plt.close("all")
            plt.show = original_show

    def test_save_delegates_to_io_adapter(self, b111_da, tmp_path) -> None:
        """MagneticMap.save() stays a thin wrapper over qdmpy.io."""
        mm = MagneticMap.from_b111(b111_da, nv_axis=(0, 0, 1), epsilon=1e-30)
        path = tmp_path / "map.nc"

        with patch("qdmpy.io.save_magnetic_map") as mock_save:
            mm.save(path)

        mock_save.assert_called_once_with(mm, path)


class TestMagneticMapPersistenceAdapter:
    """Tests for the MagneticMap persistence adapter."""

    def test_save_magnetic_map_writes_netcdf(self, tmp_path) -> None:
        from qdmpy.io import save_magnetic_map

        rng = np.random.default_rng(42)
        coords = {"y": np.arange(4), "x": np.arange(4)}
        attrs = {"pixel_spacing": 4e-6}

        def _da(name: str) -> xr.DataArray:
            return xr.DataArray(
                rng.random((4, 4)),
                dims=("y", "x"),
                coords=coords,
                attrs={**attrs, "component": name},
            )

        magnetic_map = MagneticMap(
            b111=_da("b111"),
            bx=_da("Bx"),
            by=_da("By"),
            bz=_da("Bz"),
            btotal=_da("Btotal"),
            nv_axis=(0.0, 0.0, 1.0),
        )
        path = tmp_path / "map.nc"

        save_magnetic_map(magnetic_map, path)

        assert path.exists()
        dataset = xr.load_dataset(path)
        try:
            assert set(dataset.data_vars) == {"b111", "Bx", "By", "Bz", "Btotal"}
        finally:
            dataset.close()
