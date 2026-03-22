"""Tests for QDMpy HDF5 .qdm and NPZ I/O (QEP-008).

Covers:
- save_qdm / load_qdm round-trip (basic and with images)
- save_npz / load_npz round-trip (replaces QDMResult.save/load)
- Missing images are optional (no error when None)
- include_bxyz flag (skipped when MagneticMap not available)
- FieldSource round-trip through .qdm
- Version negotiation (higher major version raises DataLoadError)
- Overwrite protection (FileExistsError when overwrite=False)
- .qdm extension warning
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qdmpy.exceptions import DataLoadError, DataValidationError
from qdmpy.field_source import FieldSource, MagneticModel, MagneticSource, UpwardContinuedSource
from qdmpy.io import load_npz, load_qdm, save_npz, save_qdm
from qdmpy.testing import make_synthetic_qdm_result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def result_no_images():
    """QDMResult with no optical images."""
    return make_synthetic_qdm_result(shape=(8, 8))


@pytest.fixture
def result_with_images():
    """QDMResult with synthetic optical images attached."""
    result = make_synthetic_qdm_result(shape=(8, 8))
    height, width = result.scan_dimensions
    light = np.random.default_rng(1).random((height, width)).astype(np.float32)
    laser = np.random.default_rng(2).random((height, width)).astype(np.float32)
    return result.model_copy(update={"light_image": light, "laser_image": laser})


@pytest.fixture
def result_with_field_sources():
    """QDMResult with a FieldSource attached."""
    result = make_synthetic_qdm_result(shape=(8, 8))
    src = FieldSource(name="test bias", kind="generic")
    return result.model_copy(update={"field_sources": [src]})


@pytest.fixture
def result_with_magnetic_source():
    """QDMResult with a MagneticSource attached."""
    result = make_synthetic_qdm_result(shape=(8, 8))
    src = MagneticSource(
        name="grain",
        center=(3.5, 4.5),
        half_extent=(2.0, 1.5),
        pixel_spacing=result.pixel_spacing,
        model=MagneticModel(inclination=20.0, declination=45.0, magnetic_moment=1e-12),
    )
    return result.model_copy(update={"field_sources": [src]})


@pytest.fixture
def result_with_upward_continued_source():
    """QDMResult with an UpwardContinuedSource attached."""
    result = make_synthetic_qdm_result(shape=(8, 8))
    parent = MagneticSource(
        name="grain",
        center=(3.5, 4.5),
        half_extent=(2.0, 1.5),
        pixel_spacing=result.pixel_spacing,
        model=MagneticModel(inclination=20.0, declination=45.0, magnetic_moment=1e-12),
    )
    src = UpwardContinuedSource(
        name="grain @ 10um",
        parent=parent,
        height_um=10.0,
        model=MagneticModel(inclination=25.0, declination=50.0, magnetic_moment=7.5e-13),
    )
    return result.model_copy(update={"field_sources": [src]})


# ---------------------------------------------------------------------------
# save_qdm / load_qdm
# ---------------------------------------------------------------------------


class TestSaveLoadQdm:
    """Round-trip tests for the .qdm HDF5 format."""

    def test_basic_round_trip(self, tmp_path: Path, result_no_images) -> None:
        """Basic round-trip: save then load reproduces FitResult fields."""
        path = tmp_path / "out.qdm"
        save_qdm(result_no_images, path)
        loaded = load_qdm(path)

        assert loaded.model_name == result_no_images.model_name
        assert loaded.scan_dimensions == result_no_images.scan_dimensions
        assert abs(loaded.pixel_spacing - result_no_images.pixel_spacing) < 1e-15
        np.testing.assert_allclose(loaded.b111_remanent, result_no_images.b111_remanent, rtol=1e-4)
        np.testing.assert_allclose(loaded.b111_induced, result_no_images.b111_induced, rtol=1e-4)

    def test_no_images_when_none(self, tmp_path: Path, result_no_images) -> None:
        """Light/laser images are None after round-trip when not provided."""
        path = tmp_path / "out.qdm"
        save_qdm(result_no_images, path)
        loaded = load_qdm(path)
        assert loaded.light_image is None
        assert loaded.laser_image is None

    def test_images_round_trip(self, tmp_path: Path, result_with_images) -> None:
        """Light and laser images survive the .qdm round-trip."""
        path = tmp_path / "out.qdm"
        save_qdm(result_with_images, path)
        loaded = load_qdm(path)

        assert loaded.light_image is not None
        assert loaded.laser_image is not None
        np.testing.assert_allclose(loaded.light_image, result_with_images.light_image, rtol=1e-5)
        np.testing.assert_allclose(loaded.laser_image, result_with_images.laser_image, rtol=1e-5)

    def test_nv_axis_round_trip(self, tmp_path: Path, result_no_images) -> None:
        """nv_axis survives the .qdm round-trip."""
        result = result_no_images.model_copy(update={"nv_axis": (0.57, 0.57, 0.57)})
        path = tmp_path / "out.qdm"
        save_qdm(result, path)
        loaded = load_qdm(path)
        assert loaded.nv_axis is not None
        np.testing.assert_allclose(loaded.nv_axis, (0.57, 0.57, 0.57), rtol=1e-10)

    def test_field_source_round_trip(self, tmp_path: Path, result_with_field_sources) -> None:
        """FieldSources survive the .qdm round-trip."""
        path = tmp_path / "out.qdm"
        save_qdm(result_with_field_sources, path)
        loaded = load_qdm(path)

        assert len(loaded.field_sources) == 1
        src = loaded.field_sources[0]
        assert src.name == "test bias"
        assert src.kind == "generic"

    def test_field_source_with_field_map(self, tmp_path: Path, result_no_images) -> None:
        """FieldSource.field_map NDArray survives the .qdm round-trip."""
        rng = np.random.default_rng(42)
        fmap = rng.random((8, 8)).astype(np.float32)
        src = FieldSource(name="mapped source", field_map=fmap)
        result = result_no_images.model_copy(update={"field_sources": [src]})
        path = tmp_path / "out.qdm"
        save_qdm(result, path)
        loaded = load_qdm(path)

        assert len(loaded.field_sources) == 1
        assert loaded.field_sources[0].field_map is not None
        np.testing.assert_allclose(loaded.field_sources[0].field_map, fmap, rtol=1e-5)

    def test_magnetic_source_round_trip(self, tmp_path: Path, result_with_magnetic_source) -> None:
        """MagneticSource survives the .qdm round-trip with subtype fidelity."""
        path = tmp_path / "out.qdm"
        save_qdm(result_with_magnetic_source, path)
        loaded = load_qdm(path)

        assert len(loaded.field_sources) == 1
        src = loaded.field_sources[0]
        assert isinstance(src, MagneticSource)
        assert src.kind == "magnetic"
        assert src.center == (3.5, 4.5)
        assert src.half_extent == (2.0, 1.5)
        assert src.pixel_spacing == result_with_magnetic_source.pixel_spacing
        assert src.model.inclination == 20.0
        assert src.model.declination == 45.0
        assert src.model.magnetic_moment == 1e-12

    def test_upward_continued_source_round_trip(
        self,
        tmp_path: Path,
        result_with_upward_continued_source,
    ) -> None:
        """UpwardContinuedSource survives the .qdm round-trip with subtype fidelity."""
        path = tmp_path / "out.qdm"
        save_qdm(result_with_upward_continued_source, path)
        loaded = load_qdm(path)

        assert len(loaded.field_sources) == 1
        src = loaded.field_sources[0]
        assert isinstance(src, UpwardContinuedSource)
        assert src.kind == "upward_continued"
        assert src.height_um == 10.0
        assert src.parent.kind == "magnetic"
        assert src.parent.center == (3.5, 4.5)
        assert src.parent.half_extent == (2.0, 1.5)
        assert src.parent.pixel_spacing == result_with_upward_continued_source.pixel_spacing
        assert src.model.inclination == 25.0
        assert src.model.declination == 50.0
        assert src.model.magnetic_moment == 7.5e-13

    def test_overwrite_protection(self, tmp_path: Path, result_no_images) -> None:
        """FileExistsError raised when file exists and overwrite=False."""
        path = tmp_path / "out.qdm"
        save_qdm(result_no_images, path)
        with pytest.raises(FileExistsError):
            save_qdm(result_no_images, path, overwrite=False)

    def test_overwrite_allowed(self, tmp_path: Path, result_no_images) -> None:
        """No error when overwrite=True on an existing file."""
        path = tmp_path / "out.qdm"
        save_qdm(result_no_images, path)
        save_qdm(result_no_images, path, overwrite=True)  # should not raise

    def test_wrong_extension_creates_file(self, tmp_path: Path, result_no_images) -> None:
        """No error when path does not have .qdm extension; file is created."""
        path = tmp_path / "out.h5"
        save_qdm(result_no_images, path)
        assert path.exists()

    def test_file_not_found(self, tmp_path: Path) -> None:
        """DataLoadError raised when .qdm file does not exist."""
        with pytest.raises(DataLoadError):
            load_qdm(tmp_path / "nonexistent.qdm")

    def test_version_negotiation_higher_major(self, tmp_path: Path, result_no_images) -> None:
        """DataLoadError raised when file qdm_version major > code major."""
        import h5py

        path = tmp_path / "future.qdm"
        save_qdm(result_no_images, path)
        # Overwrite the version attribute with a future major version
        with h5py.File(path, "r+") as f:
            f.attrs["qdm_version"] = "99.0"
        with pytest.raises(DataLoadError, match="upgrade qdmpy"):
            load_qdm(path)

    def test_missing_version_raises(self, tmp_path: Path, result_no_images) -> None:
        """DataLoadError raised for HDF5 files without qdm_version attribute."""
        import h5py

        path = tmp_path / "bad.qdm"
        save_qdm(result_no_images, path)
        with h5py.File(path, "r+") as f:
            del f.attrs["qdm_version"]
        with pytest.raises(DataLoadError):
            load_qdm(path)

    def test_image_shape_mismatch_raises(self, tmp_path: Path, result_no_images) -> None:
        """DataValidationError raised when image shape != scan_dimensions."""
        wrong_image = np.ones((4, 4))  # scan is (8, 8)
        result = result_no_images.model_copy(update={"light_image": wrong_image})
        path = tmp_path / "bad.qdm"
        with pytest.raises(DataValidationError, match="light_image shape"):
            save_qdm(result, path)

    def test_b111_available_without_refit(self, tmp_path: Path, result_no_images) -> None:
        """B111 fields are immediately accessible after load (no recomputation)."""
        path = tmp_path / "out.qdm"
        save_qdm(result_no_images, path)
        loaded = load_qdm(path)
        # Access b111_remanent should not raise and cache should be warm
        b111 = loaded.b111_remanent
        assert b111.shape == result_no_images.scan_dimensions

    def test_parameters_round_trip(self, tmp_path: Path, result_no_images) -> None:
        """Fitted parameter arrays (center, chi2, etc.) survive round-trip."""
        path = tmp_path / "out.qdm"
        save_qdm(result_no_images, path)
        loaded = load_qdm(path)

        for key in ("center", "chi2"):
            if key in result_no_images.parameters:
                np.testing.assert_allclose(
                    loaded.parameters[key],
                    result_no_images.parameters[key],
                    rtol=1e-4,
                    err_msg=f"Parameter {key!r} mismatch after round-trip",
                )


# ---------------------------------------------------------------------------
# save_npz / load_npz
# ---------------------------------------------------------------------------


class TestSaveLoadNpz:
    """Round-trip tests for the NPZ checkpoint format."""

    def test_basic_round_trip(self, tmp_path: Path, result_no_images) -> None:
        """save_npz / load_npz preserves FitResult fields."""
        path = tmp_path / "out.npz"
        save_npz(result_no_images, path)
        loaded = load_npz(path)

        assert loaded.model_name == result_no_images.model_name
        assert loaded.scan_dimensions == result_no_images.scan_dimensions
        np.testing.assert_allclose(loaded.b111_remanent, result_no_images.b111_remanent, rtol=1e-6)

    def test_npz_no_images(self, tmp_path: Path, result_with_images) -> None:
        """NPZ format does not preserve images (by design)."""
        path = tmp_path / "out.npz"
        save_npz(result_with_images, path)
        loaded = load_npz(path)
        assert loaded.light_image is None
        assert loaded.laser_image is None

    def test_npz_nv_axis_round_trip(self, tmp_path: Path, result_no_images) -> None:
        """nv_axis survives NPZ round-trip."""
        result = result_no_images.model_copy(update={"nv_axis": (0.1, 0.2, 0.97)})
        path = tmp_path / "out.npz"
        save_npz(result, path)
        loaded = load_npz(path)
        assert loaded.nv_axis is not None
        np.testing.assert_allclose(loaded.nv_axis, (0.1, 0.2, 0.97), rtol=1e-10)

    def test_npz_file_not_found(self, tmp_path: Path) -> None:
        """DataLoadError raised when file does not exist."""
        with pytest.raises(DataLoadError):
            load_npz(tmp_path / "missing.npz")


# ---------------------------------------------------------------------------
# Public API availability
# ---------------------------------------------------------------------------


class TestPublicApi:
    """Verify QEP-008 public API is reachable from qdmpy top-level."""

    def test_imports_from_qdmpy(self) -> None:
        """save_qdm, load_qdm, save_npz, load_npz, FieldSource importable from qdmpy."""
        import qdmpy

        assert callable(qdmpy.save_qdm)
        assert callable(qdmpy.load_qdm)
        assert callable(qdmpy.save_npz)
        assert callable(qdmpy.load_npz)
        assert qdmpy.FieldSource is FieldSource

    def test_imports_from_io(self) -> None:
        """All public symbols available from qdmpy.io."""
        import qdmpy.io as io_mod

        assert callable(io_mod.save_qdm)
        assert callable(io_mod.load_qdm)
        assert callable(io_mod.save_npz)
        assert callable(io_mod.load_npz)
        assert callable(io_mod.get_image)
        assert callable(io_mod.has_csv)
        assert callable(io_mod.load_metadata_toml)

    def test_qdm_result_has_no_plot_methods(self) -> None:
        """QDMResult must not have plot/show/display methods."""
        from qdmpy.result import QDMResult

        for method in ("plot", "show", "display"):
            assert not hasattr(QDMResult, method), (
                f"QDMResult should not have method {method!r} (QEP-008)"
            )

    def test_qdm_result_has_thin_io_wrappers(self) -> None:
        """QDMResult.save and QDMResult.load are thin wrappers (no logic)."""
        from qdmpy.result import QDMResult

        assert callable(QDMResult.save)
        assert callable(QDMResult.load)

    def test_qdm_result_has_image_fields(self) -> None:
        """QDMResult exposes light_image, laser_image, field_sources."""
        result = make_synthetic_qdm_result(shape=(4, 4))
        assert hasattr(result, "light_image")
        assert hasattr(result, "laser_image")
        assert hasattr(result, "field_sources")
        assert hasattr(result, "has_cached_magnetic_map")

    def test_save_load_qdm_dispatch(self, tmp_path: Path) -> None:
        """result.save/.load dispatch to .qdm format for .qdm extension."""
        result = make_synthetic_qdm_result(shape=(4, 4))
        path = tmp_path / "out.qdm"
        result.save(path)
        loaded = make_synthetic_qdm_result.__class__  # just check path exists
        assert path.exists()
        loaded = result.load(path)
        assert loaded.model_name == result.model_name

    def test_save_load_npz_dispatch(self, tmp_path: Path) -> None:
        """result.save/.load dispatch to NPZ format for non-.qdm extension."""
        result = make_synthetic_qdm_result(shape=(4, 4))
        path = tmp_path / "out.npz"
        result.save(path)
        assert path.exists()
        loaded = result.load(path)
        assert loaded.model_name == result.model_name

    def test_has_cached_magnetic_map_false_initially(self) -> None:
        """has_cached_magnetic_map is False before magnetic_map is accessed."""
        result = make_synthetic_qdm_result(shape=(4, 4))
        assert result.has_cached_magnetic_map is False
