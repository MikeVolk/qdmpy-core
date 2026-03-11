"""Physical field source descriptions for QDM measurements.

A FieldSource describes a physical contributor to the measured B-field (e.g.
a ferromagnetic layer, a current-carrying wire, an applied bias coil). Each
source carries optional pre-computed spatial field maps.

Concrete subclasses (QEP-050):
  - MagneticSource: spatially localised magnetic grain/inclusion
  - UpwardContinuedSource: same source as seen at a different sensor height
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FieldSource(BaseModel):
    """Base class for physical sources contributing to a measured B field.

    Subclasses add the parameters specific to the source geometry and
    material (current loops, ferromagnetic layers, uniform bias fields,
    etc.).

    The ``kind`` field is the Pydantic discriminator. All subclasses must
    declare it as a Literal. The base class defaults to "generic" so that
    bare FieldSource instances can round-trip through the .qdm format.

    Attributes:
        kind: Discriminator literal identifying the source type.
        name: Human-readable label for this source.
        field_map: Optional pre-computed spatial field map (H, W) in uT.
            Excluded from JSON serialisation (stored as HDF5 dataset).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["generic"] = "generic"
    name: str
    field_map: Any | None = None  # NDArray | None -- Any avoids import at runtime


class MagneticModel(BaseModel):
    """Three-parameter description of a magnetic dipole source.

    Conventions follow pypole (dec=0 -> -Y, counterclockwise) so that
    values can be passed to pypole without conversion.

    Attributes:
        inclination: Angle of magnetisation below the horizontal plane,
            in degrees. 0 = horizontal, 90 = vertically downward,
            -90 = vertically upward. Range [-90, 90].
        declination: Azimuthal angle of the horizontal magnetisation component,
            in degrees. Counterclockwise from -Y (image south):
            dec=0 -> -Y, dec=90 -> +X (East), dec=180 -> +Y, dec=270 -> -X (West).
            Range [0, 360).
        magnetic_moment: Total magnetic moment magnitude in A*m^2. Must be > 0.
    """

    inclination: float
    declination: float
    magnetic_moment: float

    @field_validator("inclination")
    @classmethod
    def _validate_inclination(cls, v: float) -> float:
        if not -90.0 <= v <= 90.0:
            raise ValueError(f"inclination must be in [-90, 90], got {v}")
        return v

    @field_validator("declination")
    @classmethod
    def _validate_declination(cls, v: float) -> float:
        if not 0.0 <= v < 360.0:
            raise ValueError(f"declination must be in [0, 360), got {v}")
        return v

    @field_validator("magnetic_moment")
    @classmethod
    def _validate_magnetic_moment(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"magnetic_moment must be > 0, got {v}")
        return v


class MagneticSource(FieldSource):
    """A spatially localised magnetic source within the scan field of view.

    Defines a rectangular region of interest (ROI) around a centre pixel
    and carries a magnetic dipole model for that region.

    Attributes:
        kind: Discriminator literal for Pydantic union deserialisation.
        center: (x, y) position of the source centre in pixels (float
            to allow sub-pixel precision). Origin at top-left of the image,
            x increases rightward, y increases downward.
        half_extent: (dx, dy) half-width and half-height of the bounding
            rectangle in pixels. The ROI spans
            [center_x - dx, center_x + dx] x [center_y - dy, center_y + dy].
        pixel_spacing: Physical size of one pixel in metres. Stored on the
            source so it can be used independently of QDMResult.
        model: Magnetic dipole model (inclination, declination, moment).
    """

    kind: Literal["magnetic"] = "magnetic"
    center: tuple[float, float]
    half_extent: tuple[float, float]
    pixel_spacing: float
    model: MagneticModel

    @field_validator("pixel_spacing")
    @classmethod
    def _validate_pixel_spacing(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"pixel_spacing must be > 0, got {v}")
        return v

    @field_validator("half_extent")
    @classmethod
    def _validate_half_extent(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] <= 0.0 or v[1] <= 0.0:
            raise ValueError(f"half_extent components must be > 0, got {v}")
        return v

    @property
    def center_um(self) -> tuple[float, float]:
        """Source centre in micrometres."""
        scale = self.pixel_spacing * 1e6
        return (self.center[0] * scale, self.center[1] * scale)

    @property
    def half_extent_um(self) -> tuple[float, float]:
        """Bounding-box half-extents in micrometres."""
        scale = self.pixel_spacing * 1e6
        return (self.half_extent[0] * scale, self.half_extent[1] * scale)

    @property
    def roi_pixels(self) -> tuple[slice, slice]:
        """NumPy index slices (row_slice, col_slice) for the ROI.

        Note: NumPy indexes as [y, x], so row = y, col = x.
        """
        cx, cy = self.center
        dx, dy = self.half_extent
        row = slice(round(cy - dy), round(cy + dy) + 1)
        col = slice(round(cx - dx), round(cx + dx) + 1)
        return row, col


class UpwardContinuedSource(FieldSource):
    """A MagneticSource as modelled at a different sensor-sample separation.

    Upward continuation changes the apparent magnetic moment and field
    distribution. This source carries its own MagneticModel representing
    the effective parameters at the continued height, while delegating all
    spatial information (center, half_extent, pixel_spacing) to the parent.

    Attributes:
        kind: Discriminator literal.
        parent: The original MagneticSource from which this source is derived.
        height_um: Additional sensor-sample separation added by upward
            continuation, in micrometres (must be > 0).
        model: Effective magnetic model at the continued height. Distinct
            from parent.model to capture the change in apparent parameters.
    """

    kind: Literal["upward_continued"] = "upward_continued"
    parent: MagneticSource
    height_um: float
    model: MagneticModel

    @field_validator("height_um")
    @classmethod
    def _validate_height_um(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"height_um must be > 0, got {v}")
        return v

    @property
    def center(self) -> tuple[float, float]:
        """Delegates to parent.center (pixels)."""
        return self.parent.center

    @property
    def half_extent(self) -> tuple[float, float]:
        """Delegates to parent.half_extent (pixels)."""
        return self.parent.half_extent

    @property
    def pixel_spacing(self) -> float:
        """Delegates to parent.pixel_spacing (metres per pixel)."""
        return self.parent.pixel_spacing

    @property
    def center_um(self) -> tuple[float, float]:
        """Delegates to parent.center_um."""
        return self.parent.center_um

    @property
    def half_extent_um(self) -> tuple[float, float]:
        """Delegates to parent.half_extent_um."""
        return self.parent.half_extent_um

    @property
    def roi_pixels(self) -> tuple[slice, slice]:
        """Delegates to parent.roi_pixels."""
        return self.parent.roi_pixels


# Discriminated union -- Pydantic selects the correct subclass on
# model_validate() based on the "kind" field.
FieldSourceType = Annotated[
    MagneticSource | UpwardContinuedSource | FieldSource,
    Field(discriminator="kind"),
]
