# QEP-008: Export System

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-041, QEP-CORE-001 |
| **Blocks** | QEP-050 |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |
| **Amended** | 2026-03-04 |
| **Implemented** | 2026-03-22 |

---

## Summary of Amendment

The original QEP-008 (2026-02-15) described NPZ and MATLAB exporters for
`FitResult`. This amendment supersedes that design:

1. Introduces **`.qdm`** -- a single HDF5 file carrying everything needed after
   fitting: optical images, fitted parameters, B111 field maps, optional Bxyz
   reconstruction, metadata, and field sources.
2. Adds `light_image` / `laser_image` optional fields to `QDMResult` so the
   images travel with the result from `Measurement.fit_odmr()` onwards.
3. Adds `field_sources: list[FieldSourceType]` to `QDMResult` -- a list of
   physical sources contributing to the measured field. `FieldSource` is a
   `BaseModel` base class with a `kind` discriminator; the concrete subclass
   taxonomy is specified in QEP-050.
4. Promotes `io.py` to an `io/` package: `io/images.py` (existing image
   loading), `io/qdm.py` (new HDF5 `.qdm` format), `io/npz.py` (existing NPZ
   checkpoint format). All I/O lives here, not on `QDMResult`.
5. Removes `plot()`, `show()`, `display()`, `save()`, `load()` from
   `QDMResult`, making it a pure data container. Plotting functions in
   `plotting.py` accept `QDMResult` directly.

Status of original sub-proposals:

| Original section | Disposition |
|-----------------|-------------|
| `NpzExporter` / `MatlabExporter` / `BaseExporter` | Dropped -- no pluggable exporter hierarchy needed |
| `export_results()` dispatch function | Dropped -- replaced by `save_qdm()` |
| `FitResult.load_results()` LSP fix | Done in QEP-CORE-001 |
| Field conversion utilities (`convert.py`) | Deferred -- `MagneticMap` covers Bxyz |

---

## Motivation

After `meas.fit_odmr()` returns a `QDMResult`, a user typically wants to:

- Archive the full measurement result for later re-analysis.
- Share it with collaborators who may use MATLAB, Julia, or another Python
  environment -- they must be able to read it without qdmpy installed.
- Reproduce every figure in a paper from a single file.
- Reload it in a future session without re-fitting (~minutes on a 2k x 2k scan).

The current NPZ format stores only the fitted parameters. It does not include:

- Optical images (`light_image`, `laser_image`/diamond) needed to interpret maps.
- Physical field quantities (B111, Bxyz) in labelled, unit-tagged form.
- Enough self-description to open the file without qdmpy's schema knowledge.

---

## Requirements

1. **Single-file export** -- one `.qdm` file reproduces every standard plot.
2. **Self-describing** -- key names, units, and dimensions are embedded in the
   file, not in qdmpy source code.
3. **Pickle-free** -- no `allow_pickle=True` anywhere (QEP-CORE-001).
4. **Round-trip** -- `load_qdm(path)` reloads a functional `QDMResult`.
5. **Backward compatibility** -- existing NPZ round-trip preserved as
   `save_npz()` / `load_npz()` in the same io module.
6. **Optional images** -- export succeeds when `light_image`/`laser_image` are
   `None`.
7. **Optional Bxyz** -- MagneticMap reconstruction is expensive; Bxyz is
   included only when the caller explicitly opts in.
8. **Field sources** -- `QDMResult` carries a list of `FieldSource` objects
   describing physical contributions to the measured field; these round-trip
   through `.qdm`.
9. **QDMResult is a pure data container** -- no I/O methods, no plotting
   methods. I/O lives in `io/qdm.py`, plotting lives in `plotting.py`.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## File Format: HDF5 (`.qdm`)

The `.qdm` format is an HDF5 file (extension `.qdm`, magic bytes `\x89HDF`).

**Why HDF5?**

| Property | HDF5 | ZIP+NPZ | NetCDF |
|----------|------|---------|--------|
| Self-describing groups/attrs | yes | no | yes |
| Browsable without qdmpy | HDFView / h5py | no | ncdump |
| Interoperable (MATLAB, Julia) | yes | no | partial |
| Per-dataset compression | yes | yes (whole) | yes |
| Dependency | h5py (2 MB wheel) | stdlib | netcdf4 or h5netcdf |
| Heterogeneous arrays (no coordinate requirement) | yes | yes | awkward |

NetCDF is built on HDF5 but requires all arrays to have named coordinate
dimensions -- awkward for optical images and raw fit parameter arrays that lack
meaningful axis labels. ZIP+NPZ is self-contained but not self-describing and
not readable from MATLAB without a bespoke reader. HDF5 offers the best
interoperability for a published archival format.

### Layout

```
<name>.qdm   (HDF5 root)
|
+-- .attrs
|   +-- qdm_version    str       "1.0"
|   +-- model_name     str       e.g. "ESR14N"
|   +-- pixel_spacing  float     metres, e.g. 4e-6  (authoritative source)
|   +-- scan_dimensions int[2]   (height, width) in pixels
|   +-- nv_axis        float[3]  (absent if None)
|   +-- created_at     str       ISO 8601 UTC timestamp
|   +-- metadata       str       JSON-encoded dict from FitResult.metadata
|
+-- images/                        (group; omitted if both images are None)
|   +-- light          (H, W)  float32   .attrs["units"] = "a.u."
|   +-- laser          (H, W)  float32   .attrs["units"] = "a.u."
|                               (also called "diamond image")
|
+-- fit/                           (group; always present)
|   +-- frequencies    (n_frange, n_freq)  float64  .attrs["units"] = "GHz"
|   +-- <param_name>   (n_pol, n_frange, H, W) or subset  float32
|   |     for each key in FitResult.parameters
|   |     .attrs["dims"] = comma-separated dim labels
|   +-- fit_states     int32   convergence codes from gpufit
|
+-- b_field/                       (group; always present)
|   +-- b111_remanent  (H, W)  float32  .attrs["units"] = "uT"
|   +-- b111_induced   (H, W)  float32  .attrs["units"] = "uT"
|   +-- bx             (H, W)  float32  .attrs["units"] = "uT"  (optional)
|   +-- by             (H, W)  float32  .attrs["units"] = "uT"  (optional)
|   +-- bz             (H, W)  float32  .attrs["units"] = "uT"  (optional)
|   +-- btotal         (H, W)  float32  .attrs["units"] = "uT"  (optional)
|
+-- field_sources/                 (group; omitted if list is empty)
    +-- .attrs["count"]  int   number of sources
    +-- 0/                         (one subgroup per source, zero-indexed)
    |   +-- .attrs["kind"]  str    discriminator value, e.g. "magnetic"
    |   +-- .attrs["json"]  str    model_dump(mode="json", exclude={"field_map"})
    |   +-- field_map       (H, W) float32  .attrs["units"] = "uT"  (optional)
    +-- 1/ ...
    +-- N/ ...
```

Notes:
- Bxyz datasets are written only when `include_bxyz=True`.
- `pixel_spacing` in root attrs is the authoritative source; the value in
  `FitResult.metadata` is ignored on load to avoid drift.
- `created_at` is an ISO 8601 UTC timestamp set at export time.
- `fit/frequencies` stores the ODMR sweep frequencies so that fit curves can
  be overlaid on spectra without the original measurement data.
- Each `FieldSource` is serialised as JSON via
  `model_dump(mode="json", exclude={"field_map"})`. The `field_map` NDArray
  is stored as a sibling HDF5 dataset (not in JSON, since NDArrays are not
  JSON-serialisable). The `kind` attr is the Pydantic discriminator used by
  `TypeAdapter(FieldSourceType).validate_json()` on load.

### Version Negotiation

`load_qdm()` checks `qdm_version`:

- **Same major version** (e.g. file is "1.2", code knows "1.0"): load
  succeeds. Unknown groups/datasets are ignored with a debug log.
- **Higher major version** (e.g. file is "2.0", code knows "1.x"): raise
  `DataLoadError` with a message asking the user to upgrade qdmpy.
- **Missing `qdm_version`**: raise `DataLoadError` (not a valid `.qdm` file).

---

## API Changes

### 1. `FieldSource` base class (new module `src/qdmpy/field_source.py`)

```python
class FieldSource(BaseModel):
    """Base class for physical sources contributing to a measured B field.

    Subclasses add the parameters specific to the source geometry and
    material (current loops, ferromagnetic layers, uniform bias fields,
    etc.). The full subclass taxonomy is defined in QEP-050.

    The ``kind`` field is the Pydantic discriminator. All subclasses must
    declare it as a Literal. The base class defaults to "generic" so that
    bare FieldSource instances can participate in the discriminated union.

    Attributes:
        kind: Discriminator literal identifying the source type.
        name: Human-readable label for this source.
        field_map: Optional pre-computed spatial field map (H, W) in uT.
            Excluded from JSON serialisation (stored as HDF5 dataset).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "generic"
    name: str
    field_map: NDArray | None = None
```

### 2. `QDMResult` becomes a pure data container

```python
class QDMResult(BaseModel):
    """Unified result container from a single QDM measurement.

    Pure data container. All I/O is handled by ``qdmpy.io``.
    All plotting is handled by ``qdmpy.plotting``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fit_result: FitResult
    nv_axis: tuple[float, float, float] | None = None
    reconstructor: Any | None = None
    light_image: NDArray | None = None        # LED reflectance image
    laser_image: NDArray | None = None        # NV fluorescence ("diamond image")
    field_sources: list[FieldSourceType] = []  # physical B-field sources

    # --- delegated properties (unchanged) ---
    # scan_dimensions, pixel_spacing, model_name, parameters, metadata,
    # centers, linewidths, contrasts, offsets, chi2, fit_states,
    # b111, b111_remanent, b111_induced,
    # get_parameter(), get_parameter_map(), get_fit_quality_metrics(),
    # calculate_b_field()

    # --- lazy derived data ---
    @property
    def magnetic_map(self) -> MagneticMap: ...

    @property
    def has_cached_magnetic_map(self) -> bool:
        """Whether the MagneticMap has already been computed."""
        return self._magnetic_map_cache is not None

    # REMOVED: plot(), show(), display(), save(), load()
```

`Measurement.fit_odmr()` passes images automatically:

```python
# Inside Measurement.fit_odmr():
return QDMResult(
    fit_result=fit_result,
    nv_axis=...,
    light_image=self.light_image,   # same array, not copied
    laser_image=self.laser_image,
)
```

### 3. `io/` package -- all I/O as free functions

The existing `src/qdmpy/io.py` is promoted to a package:

```
src/qdmpy/io/
    __init__.py     # re-exports public API
    images.py       # get_image(), get_image_file(), load_metadata_toml()
                    #   (moved from the old io.py, unchanged)
    qdm.py          # save_qdm(), load_qdm()
    npz.py          # save_npz(), load_npz()
                    #   (moved from QDMResult.save() / .load())
```

`__init__.py` re-exports everything so that `from qdmpy.io import save_qdm`
works, and existing `from qdmpy.io import get_image` remains valid:

```python
# src/qdmpy/io/__init__.py
from qdmpy.io.images import get_image, get_image_file, has_csv, load_metadata_toml
from qdmpy.io.npz import load_npz, save_npz
from qdmpy.io.qdm import load_qdm, save_qdm
```

#### `io/qdm.py`

```python
def save_qdm(
    result: QDMResult,
    path: str | PathLike,
    *,
    include_bxyz: bool = False,
    overwrite: bool = False,
    compress: bool = True,
) -> None:
    """Export a QDMResult to a .qdm (HDF5) file.

    Args:
        result: The QDMResult to export.
        path: Destination file path. Warns if extension is not '.qdm'.
        include_bxyz: If True, compute MagneticMap (if not cached) and
            store Bx/By/Bz/Btotal. Default False.
        overwrite: If True, overwrite existing file. If False (default),
            raise FileExistsError when the file already exists.
        compress: Apply GZIP compression to each dataset.
    """


def load_qdm(path: str | PathLike) -> QDMResult:
    """Load a QDMResult from a .qdm (HDF5) file.

    Reconstructs FitResult, optional images, field_sources, and nv_axis.
    B111 fields stored in the file are loaded into FitResult caches so
    they are available immediately without recomputation. MagneticMap is
    NOT reconstructed on load -- access .magnetic_map to trigger it.

    Args:
        path: Path to a .qdm file created by save_qdm().

    Returns:
        QDMResult with all fields present in the file populated.

    Raises:
        DataLoadError: File not found, not a valid .qdm file, or
            incompatible major version.
    """
```

#### `io/npz.py`

```python
def save_npz(result: QDMResult, path: str | PathLike) -> None:
    """Save QDMResult to a pickle-free NPZ file (fit data only).

    Lightweight checkpoint format. Does not include images, Bxyz,
    or field_sources. Logic moved from QDMResult.save().
    """


def load_npz(path: str | PathLike) -> QDMResult:
    """Load QDMResult from a pickle-free NPZ file.

    Logic moved from QDMResult.load().
    """
```

### 4. Plotting accepts QDMResult directly

The existing plotting functions in `plotting.py` are updated to accept
`QDMResult` and use `result.light_image` / `result.laser_image` when
available, falling back to the `measurement` argument when images are None:

```python
# plotting.py
def plot_qdm_display(
    result: QDMResult,
    *,
    measurement: Measurement | None = None,
) -> None:
    """Display comprehensive overview.

    Uses result.light_image / result.laser_image if available.
    Falls back to measurement.light_image / measurement.laser_image
    if provided.
    """
```

---

## Usage Examples

### Typical flow

```python
from qdmpy.io import save_qdm

meas = Measurement.from_folder(path)
meas.odmr.process_data()

result = meas.fit_odmr()              # QDMResult with images attached
save_qdm(result, "results/run_001.qdm")

# Force 3D field reconstruction and include:
save_qdm(result, "results/run_001_full.qdm", include_bxyz=True)
```

### Round-trip

```python
from qdmpy.io import load_qdm

result = load_qdm("results/run_001.qdm")
result.b111_remanent    # NDArray (H, W), uT -- loaded from file, no recomputation
result.light_image      # NDArray (H, W)
result.magnetic_map.bz  # recomputed on access (Fourier reconstruction)
```

### Plotting (no methods on QDMResult)

```python
from qdmpy.io import load_qdm
from qdmpy.plotting import plot_qdm_display

result = load_qdm("results/run_001.qdm")
plot_qdm_display(result)   # uses result.light_image directly
```

### Manual construction (User 3 / custom pipeline)

```python
from qdmpy.io import save_qdm

fit_result = FitManager("ESR14N").fit(odmr_data, freq)
result = QDMResult(
    fit_result=fit_result,
    light_image=led_array,
    laser_image=nv_array,
)
save_qdm(result, "custom_run.qdm")
```

### Adding field sources before export

```python
from qdmpy.io import save_qdm

result = meas.fit_odmr()

bias = UniformBiasSource(name="applied bias", bz_uT=45.0)
result = result.model_copy(update={"field_sources": [bias]})
save_qdm(result, "results/run_001.qdm")
```

### Reloading with field sources

```python
from qdmpy.io import load_qdm

result = load_qdm("results/run_001.qdm")
for src in result.field_sources:
    print(src.name, type(src).__name__)
```

### Interoperability (reading from MATLAB)

```matlab
h5info("run_001.qdm")            % list groups/datasets
b111 = h5read("run_001.qdm", "/b_field/b111_remanent");
light = h5read("run_001.qdm", "/images/light");
freq = h5read("run_001.qdm", "/fit/frequencies");
```

### Check Bxyz cache before exporting

```python
# Explicit -- no hidden tri-state behaviour
save_qdm(result, "out.qdm", include_bxyz=result.has_cached_magnetic_map)
```

---

## Dependency

`h5py` is added as a **required** runtime dependency.

Rationale: making it optional means `save_qdm()` raises `ImportError` in
the default install -- an inexplicable failure for the most common user action.
h5py is a 2 MB pre-compiled wheel available for all CPython versions qdmpy
supports (3.12+) on Linux, macOS, and Windows. The cost is negligible.

```toml
# pyproject.toml
dependencies = [
    ...
    "h5py>=3.10",
]
```

---

## Files to Change

| File | Change |
|------|--------|
| `src/qdmpy/field_source.py` | New -- `FieldSource` base class with `kind` discriminator |
| `src/qdmpy/io/__init__.py` | New -- re-exports from submodules; replaces old `io.py` |
| `src/qdmpy/io/images.py` | Move from old `io.py` -- `get_image()`, `load_metadata_toml()`, etc. (unchanged logic) |
| `src/qdmpy/io/qdm.py` | New -- `save_qdm()`, `load_qdm()` |
| `src/qdmpy/io/npz.py` | New -- `save_npz()`, `load_npz()` (logic moved from `QDMResult.save()`/`.load()`) |
| `src/qdmpy/io.py` | Delete -- replaced by `io/` package |
| `src/qdmpy/result.py` | Add `light_image`, `laser_image`, `field_sources`, `has_cached_magnetic_map`; remove `plot()`, `show()`, `display()`, `save()`, `load()` |
| `src/qdmpy/measurement.py` | Pass `light_image`, `laser_image` to `QDMResult` in `fit_odmr()` |
| `src/qdmpy/plotting.py` | Update `plot_qdm_display()` to read images from `QDMResult` first, fall back to `measurement` |
| `pyproject.toml` | Add `h5py>=3.10` |
| `src/qdmpy/__init__.py` | Export `FieldSource`, `save_qdm`, `load_qdm`; update `io` imports |
| `tests/test_io_qdm.py` | New -- round-trip `.qdm`, round-trip `.npz`, missing images, `include_bxyz`, field_sources, version negotiation, overwrite protection |
| `tests/test_result.py` | Update -- remove tests for deleted methods |

---

## Rejected Alternatives

**ZIP+NPZ with namespaced keys (`fit__chi2`, `b_field__b111_remanent`)**:
No new dependency, but not self-describing, not browsable from MATLAB/Julia
without a custom reader, and the `__` key convention is fragile.

**NetCDF via xarray `to_netcdf()`**: Built on HDF5 but requires every array to
have named coordinate dimensions. Optical images and raw fit-parameter arrays
have no meaningful axis coordinates; inventing them adds noise.

**Make h5py an optional `[export]` extra**: Tempting for minimal base install,
but a core user action failing silently until an optional group is installed is
a worse experience than the 2 MB wheel.

**Keep NPZ for images too, rename to `.qdm`**: No self-description, flat key
namespace, not browsable with standard tools.

**`include_bxyz: bool | None = None` tri-state**: The `None` means "include if
already cached", which makes the same `save_qdm()` call produce different output
files depending on whether `.magnetic_map` was accessed earlier in the session.
Non-deterministic. Replaced with a simple `bool = False` and
`result.has_cached_magnetic_map` property for explicit opt-in.

**I/O methods on QDMResult**: `result.export()`, `result.save()`, etc. violate
Single Responsibility -- `QDMResult` should be a data container, not an I/O
controller. Free functions in `io/qdm.py` keep serialisation details out of the
data model and allow testing I/O independently.

**Plotting methods on QDMResult**: `result.plot()`, `result.show()`,
`result.display()` are thin wrappers that add no value over calling the
plotting functions directly. Removing them keeps `QDMResult` focused as a data
container and avoids coupling it to matplotlib.

---

## Resolved Questions (from architect review)

1. **`laser_image` vs `diamond_image` naming**: keep `laser_image` in code
   (matches file keyword). Documentation and tutorials may refer to it as
   "diamond image" or "NV fluorescence image".

2. **B111 in file vs recomputed**: store B111 as datasets AND populate
   FitResult caches on load. User expects instant access without refit.

3. **`FieldSource` base lacks `kind`**: resolved -- base class now has
   `kind: str = "generic"` so it can participate in the discriminated union.

4. **HDF5 `type` vs Pydantic `kind`**: unified -- use `kind` everywhere.
   The `kind` attr in HDF5 stores the same short literal as the Pydantic
   discriminator (e.g. `"magnetic"`, `"upward_continued"`). No fully-qualified
   class names in the file format.

5. **`model_dump()` crash on NDArray `field_map`**: resolved -- serialisation
   uses `model_dump(mode="json", exclude={"field_map"})`. The `field_map`
   array is stored as a sibling HDF5 dataset.

6. **`pixel_spacing` duplication**: root attr `pixel_spacing` is authoritative.
   On load, `FitResult.pixel_spacing` is populated from it.

7. **`save()` / `load()` convenience wrappers kept on `QDMResult`**: The
   proposal called for removing all I/O methods from `QDMResult`. In practice,
   thin `save()` and `load()` wrappers were retained (`result.py:201-250`)
   that dispatch to `qdmpy.io.save_qdm` / `save_npz` / `load_qdm` / `load_npz`
   based on file extension. They contain zero serialisation logic -- just
   routing. This improves ergonomics (`result.save("out.qdm")`) without
   violating SRP. The real I/O implementations remain in `io/qdm.py` and
   `io/npz.py`.

8. **Image dimension mismatch (binning)**: `save_qdm()` validates that
   `light_image` and `laser_image` shapes match `scan_dimensions` when
   present. If they do not match (e.g., original resolution vs binned fit),
   a `DataValidationError` is raised. Users should either bin the images to
   match or omit them.

---

## Open Questions

*All resolved.*

1. ~~**Scan metadata schema**: Deferred to QEP-047.~~ Resolved -- QEP-047
   implemented (2026-03-06). `metadata.toml` loading is integrated in
   `Measurement.from_folder()`.

2. ~~**`FieldSource` subclass taxonomy**: Deferred to QEP-050.~~ Resolved --
   QEP-050 implemented. `MagneticSource`, `UpwardContinuedSource`, and
   discriminated union `FieldSourceType` live in `field_source.py`.
