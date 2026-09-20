"""Tests for QEP-050 field source classes.

Covers: construction, property conversions, round-trip JSON via Pydantic,
validation errors, and discriminated-union deserialisation.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from qdmpy.field_source import (
    FieldSource,
    FieldSourceType,
    MagneticModel,
    MagneticSource,
    UpwardContinuedSource,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> MagneticModel:
    return MagneticModel(inclination=45.0, declination=180.0, magnetic_moment=1e-14)


@pytest.fixture
def source(model: MagneticModel) -> MagneticSource:
    return MagneticSource(
        name="Ni inclusion A",
        center=(120.5, 88.0),
        half_extent=(12.0, 8.0),
        pixel_spacing=4e-6,
        model=model,
    )


@pytest.fixture
def uc_source(source: MagneticSource) -> UpwardContinuedSource:
    uc_model = MagneticModel(inclination=45.0, declination=180.0, magnetic_moment=0.9e-14)
    return UpwardContinuedSource(
        name="Ni inclusion A @ 2 um",
        parent=source,
        height_um=2.0,
        model=uc_model,
    )


# ---------------------------------------------------------------------------
# MagneticModel
# ---------------------------------------------------------------------------


class TestMagneticModel:
    def test_construction(self, model: MagneticModel) -> None:
        assert model.inclination == 45.0
        assert model.declination == 180.0
        assert model.magnetic_moment == pytest.approx(1e-14)

    def test_inclination_boundary(self) -> None:
        MagneticModel(inclination=-90.0, declination=0.0, magnetic_moment=1.0)
        MagneticModel(inclination=90.0, declination=0.0, magnetic_moment=1.0)

    def test_inclination_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            MagneticModel(inclination=91.0, declination=0.0, magnetic_moment=1.0)
        with pytest.raises(ValidationError):
            MagneticModel(inclination=-91.0, declination=0.0, magnetic_moment=1.0)

    def test_declination_boundary(self) -> None:
        MagneticModel(inclination=0.0, declination=0.0, magnetic_moment=1.0)
        MagneticModel(inclination=0.0, declination=359.9, magnetic_moment=1.0)

    def test_declination_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            MagneticModel(inclination=0.0, declination=360.0, magnetic_moment=1.0)
        with pytest.raises(ValidationError):
            MagneticModel(inclination=0.0, declination=-1.0, magnetic_moment=1.0)

    def test_magnetic_moment_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            MagneticModel(inclination=0.0, declination=0.0, magnetic_moment=0.0)
        with pytest.raises(ValidationError):
            MagneticModel(inclination=0.0, declination=0.0, magnetic_moment=-1e-14)


# ---------------------------------------------------------------------------
# MagneticSource
# ---------------------------------------------------------------------------


class TestMagneticSource:
    def test_construction(self, source: MagneticSource) -> None:
        assert source.name == "Ni inclusion A"
        assert source.center == (120.5, 88.0)
        assert source.half_extent == (12.0, 8.0)
        assert source.pixel_spacing == pytest.approx(4e-6)
        assert source.kind == "magnetic"

    def test_center_um(self, source: MagneticSource) -> None:
        cx, cy = source.center_um
        assert cx == pytest.approx(120.5 * 4e-6 * 1e6)
        assert cy == pytest.approx(88.0 * 4e-6 * 1e6)

    def test_half_extent_um(self, source: MagneticSource) -> None:
        dx, dy = source.half_extent_um
        assert dx == pytest.approx(12.0 * 4e-6 * 1e6)
        assert dy == pytest.approx(8.0 * 4e-6 * 1e6)

    def test_roi_pixels(self, source: MagneticSource) -> None:
        row, col = source.roi_pixels
        # center=(120.5, 88.0), half_extent=(12.0, 8.0)
        # row: y in [88-8, 88+8] => [80, 97]
        assert row == slice(80, 97)
        # col: x in [120.5-12, 120.5+12] => [108, 133]
        assert col == slice(108, 133)

    def test_pixel_spacing_must_be_positive(self, model: MagneticModel) -> None:
        with pytest.raises(ValidationError):
            MagneticSource(
                name="x",
                center=(0.0, 0.0),
                half_extent=(1.0, 1.0),
                pixel_spacing=0.0,
                model=model,
            )
        with pytest.raises(ValidationError):
            MagneticSource(
                name="x",
                center=(0.0, 0.0),
                half_extent=(1.0, 1.0),
                pixel_spacing=-1e-6,
                model=model,
            )

    def test_half_extent_must_be_positive(self, model: MagneticModel) -> None:
        with pytest.raises(ValidationError):
            MagneticSource(
                name="x",
                center=(0.0, 0.0),
                half_extent=(0.0, 1.0),
                pixel_spacing=1e-6,
                model=model,
            )
        with pytest.raises(ValidationError):
            MagneticSource(
                name="x",
                center=(0.0, 0.0),
                half_extent=(1.0, -1.0),
                pixel_spacing=1e-6,
                model=model,
            )

    def test_json_round_trip(self, source: MagneticSource) -> None:
        json_str = source.model_dump_json()
        restored = MagneticSource.model_validate_json(json_str)
        assert restored.center == source.center
        assert restored.half_extent == source.half_extent
        assert restored.pixel_spacing == pytest.approx(source.pixel_spacing)
        assert restored.model.inclination == source.model.inclination


# ---------------------------------------------------------------------------
# UpwardContinuedSource
# ---------------------------------------------------------------------------


class TestUpwardContinuedSource:
    def test_construction(self, uc_source: UpwardContinuedSource, source: MagneticSource) -> None:
        assert uc_source.kind == "upward_continued"
        assert uc_source.height_um == pytest.approx(2.0)
        assert uc_source.model.magnetic_moment == pytest.approx(0.9e-14)

    def test_delegates_spatial_properties(
        self, uc_source: UpwardContinuedSource, source: MagneticSource
    ) -> None:
        assert uc_source.center == source.center
        assert uc_source.half_extent == source.half_extent
        assert uc_source.pixel_spacing == pytest.approx(source.pixel_spacing)
        assert uc_source.center_um == source.center_um
        assert uc_source.half_extent_um == source.half_extent_um
        assert uc_source.roi_pixels == source.roi_pixels

    def test_height_um_must_be_positive(self, source: MagneticSource) -> None:
        uc_model = MagneticModel(inclination=0.0, declination=0.0, magnetic_moment=1.0)
        with pytest.raises(ValidationError):
            UpwardContinuedSource(name="x", parent=source, height_um=0.0, model=uc_model)
        with pytest.raises(ValidationError):
            UpwardContinuedSource(name="x", parent=source, height_um=-1.0, model=uc_model)

    def test_json_round_trip(self, uc_source: UpwardContinuedSource) -> None:
        json_str = uc_source.model_dump_json()
        restored = UpwardContinuedSource.model_validate_json(json_str)
        assert restored.height_um == pytest.approx(uc_source.height_um)
        assert restored.parent.name == uc_source.parent.name
        assert restored.parent.center == uc_source.parent.center


# ---------------------------------------------------------------------------
# Discriminated union (FieldSourceType)
# ---------------------------------------------------------------------------


class TestFieldSourceType:
    def test_discriminator_selects_magnetic(self, source: MagneticSource) -> None:
        ta = TypeAdapter(FieldSourceType)
        data = source.model_dump(mode="json")
        result = ta.validate_python(data)
        assert isinstance(result, MagneticSource)
        assert result.name == source.name

    def test_discriminator_selects_upward_continued(self, uc_source: UpwardContinuedSource) -> None:
        ta = TypeAdapter(FieldSourceType)
        data = uc_source.model_dump(mode="json")
        result = ta.validate_python(data)
        assert isinstance(result, UpwardContinuedSource)
        assert result.height_um == pytest.approx(uc_source.height_um)

    def test_discriminator_selects_generic(self) -> None:
        ta = TypeAdapter(FieldSourceType)
        data = {"kind": "generic", "name": "bare source"}
        result = ta.validate_python(data)
        assert isinstance(result, FieldSource)
        assert result.kind == "generic"

    def test_json_round_trip_via_union(self, uc_source: UpwardContinuedSource) -> None:
        ta = TypeAdapter(FieldSourceType)
        json_str = uc_source.model_dump_json()
        result = ta.validate_json(json_str)
        assert isinstance(result, UpwardContinuedSource)
        assert result.parent.name == uc_source.parent.name
