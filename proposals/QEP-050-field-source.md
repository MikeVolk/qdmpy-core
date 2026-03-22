# QEP-050 — FieldSource: Physical Magnetic Source Model

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | S |
| **Depends on** | QEP-008 (amended), QEP-041 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-04 |

---

## Motivation

QEP-008 (amended) introduced `FieldSource` as a `BaseModel` base class on
`QDMResult`, with round-trip serialisation through the `.qdm` HDF5 format.
The base class holds only `name` and an optional `field_map`. This QEP
defines the first two concrete subclasses:

- **`MagneticSource`** — a spatially localised magnetic source (inclusion,
  grain, thin-film feature) described by its position, bounding extent, and
  a three-parameter magnetic model.
- **`UpwardContinuedSource`** — the same source as seen at a different
  sensor height, with its own effective magnetic model.

These two types cover the dominant QDM analysis workflow: identify a
magnetic feature, characterise its dipole-like parameters, then model how
those parameters change when the sensor–sample distance changes (upward
continuation).

---

## Design

### `MagneticModel` — sub-model for magnetic properties

```python
class MagneticModel(BaseModel):
    """Three-parameter description of a magnetic dipole source.

    Attributes:
        inclination: Angle of magnetisation below the horizontal plane,
            in degrees. 0 = horizontal, 90 = vertically downward,
            -90 = vertically upward.
        declination: Azimuthal angle of the horizontal magnetisation
            component measured clockwise from the +y axis (image north),
            in degrees. 0 = +y, 90 = +x.
        magnetic_moment: Total magnetic moment magnitude in A*m^2.
    """

    inclination: float      # degrees, range [-90, 90]; 0 = horizontal, 90 = down
    declination: float      # degrees, range [0, 360); pypole convention: 0 = -Y (counterclockwise)
    magnetic_moment: float  # A*m^2, must be > 0
```

### `MagneticSource` — localised source

```python
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
    pixel_spacing: float        # metres per pixel, > 0
    model: MagneticModel

    # ------------------------------------------------------------------
    # Convenience properties (physical units)
    # ------------------------------------------------------------------

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
        row = slice(int(round(cy - dy)), int(round(cy + dy)) + 1)
        col = slice(int(round(cx - dx)), int(round(cx + dx)) + 1)
        return row, col
```

### `UpwardContinuedSource` — source at a different sensor height

```python
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
    height_um: float            # micrometres, > 0
    model: MagneticModel

    # ------------------------------------------------------------------
    # Delegated spatial properties
    # ------------------------------------------------------------------

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
```

---

## Discriminated Union for `QDMResult`

Both classes use a `kind` literal field. `QDMResult.field_sources` is typed
as a discriminated union so Pydantic selects the correct subclass on
`model_validate()`:

```python
# src/qdmpy/field_source.py
from typing import Annotated
from pydantic import Field

FieldSourceType = Annotated[
    MagneticSource | UpwardContinuedSource,
    Field(discriminator="kind"),
]
```

```python
# src/qdmpy/result.py
from qdmpy.field_source import FieldSource, FieldSourceType

class QDMResult(BaseModel):
    ...
    field_sources: list[FieldSourceType] = []
```

Adding a new source type in the future only requires:
1. Defining the subclass with a unique `kind` literal in `field_source.py`.
2. Adding it to the `FieldSourceType` union.
3. No changes to `QDMResult`, `export()`, or `from_qdm()`.

---

## Serialisation in `.qdm` (HDF5)

Each source is stored under `field_sources/<index>/`:

```
field_sources/
├── 0/
│   ├── .attrs["kind"]  = "magnetic"
│   ├── .attrs["json"]  = '{"kind":"magnetic","name":"Ni grain","center":[120.5,88.0],...}'
│   └── field_map       (H, W) float32  (if present)
└── 1/
    ├── .attrs["kind"]  = "upward_continued"
    ├── .attrs["json"]  = '{"kind":"upward_continued","name":"Ni grain @ 2um","parent":{...},...}'
    └── field_map       (H, W) float32  (if present)
```

`json` contains the full `model_dump(mode="json")` output, which for
`UpwardContinuedSource` includes the nested `parent` dict. On load,
`TypeAdapter(FieldSourceType).validate_json(attrs["json"])` reconstructs
the correct subclass.

---

## Usage Examples

```python
from qdmpy.field_source import MagneticModel, MagneticSource, UpwardContinuedSource

model = MagneticModel(inclination=45.0, declination=180.0, magnetic_moment=1e-14)

source = MagneticSource(
    name="Ni inclusion A",
    center=(120.5, 88.0),      # pixels
    half_extent=(12.0, 8.0),   # pixels
    pixel_spacing=4e-6,        # 4 µm/pixel
    model=model,
)

# Spatial convenience:
source.center_um          # (482.0, 352.0) µm
source.half_extent_um     # (48.0, 32.0) µm
source.roi_pixels         # (slice(80, 97), slice(108, 133))

# Upward-continued version:
uc_model = MagneticModel(inclination=45.0, declination=180.0, magnetic_moment=0.9e-14)
uc_source = UpwardContinuedSource(
    name="Ni inclusion A @ 2 µm",
    parent=source,
    height_um=2.0,
    model=uc_model,
)

# Attach to result and export:
result = result.model_copy(update={"field_sources": [source, uc_source]})
result.export("run_001.qdm")

# Round-trip:
loaded = QDMResult.from_qdm("run_001.qdm")
src = loaded.field_sources[0]     # MagneticSource
uc  = loaded.field_sources[1]     # UpwardContinuedSource
uc.parent.name                    # "Ni inclusion A"
uc.center                         # (120.5, 88.0) — delegated from parent
```

---

## Validation

`MagneticModel` and `MagneticSource` use Pydantic `field_validator` for:

- `inclination` in `[-90, 90]`
- `declination` in `[0, 360)`
- `magnetic_moment > 0`
- `pixel_spacing > 0`
- `half_extent` components `> 0`
- `UpwardContinuedSource.height_um > 0`

---

## Files to Create / Change

| File | Change |
|------|--------|
| `src/qdmpy/field_source.py` | New — `MagneticModel`, `MagneticSource`, `UpwardContinuedSource`, `FieldSourceType` |
| `src/qdmpy/result.py` | Change `field_sources` type from `list[FieldSource]` to `list[FieldSourceType]` |
| `src/qdmpy/__init__.py` | Export `MagneticSource`, `UpwardContinuedSource`, `MagneticModel`, `FieldSource` |
| `tests/test_field_source.py` | New — construction, property conversions, round-trip JSON, Pydantic validation errors |

---

## Rejected Alternatives

**Inheritance instead of composition for `UpwardContinuedSource`**:
`UpwardContinuedSource(MagneticSource)` and adding `height_um`. Rejected
because the upward-continued source is not *a kind of* `MagneticSource` —
it is a *view of* one. Composition (`parent:`) makes the relationship
explicit and prevents accidental mutation of the parent's spatial fields.

**Single class with `height_um: float | None`**: A flag argument that
changes the semantics of `model` depending on whether height is set.
Rejected — violates single-responsibility and makes the type ambiguous.

**Store only `inclination` and `declination` as a unit vector**: More
compact but loses the `magnetic_moment` magnitude, which is the primary
physical observable from QDM fitting.

---

## Open Questions

1. **Field computation** *(resolved)*: `source_fitting.compute_field(source, standoff_m)`
   evaluates the analytical dipole forward model over the source ROI and returns
   the predicted Bz in Tesla. Subtract from `bz_map_T[source.roi_pixels]` for
   residual analysis.

2. **Multiple polarities / heights**: `UpwardContinuedSource` currently
   stores a single `height_um`. If multiple continuation heights are needed
   (e.g., a depth profile), a list of `UpwardContinuedSource` objects should
   be used rather than extending this class.

3. **Coordinate origin convention** *(closed — won't fix)*: `center` stays in
   image-convention pixels (top-left origin, y-down). A y-flip for MATLAB
   QDMlab compatibility is a plotting/export concern and should be handled at
   that boundary, not in the source model.
