# Migration Guide: QDMpy -> qdmpy-core

This guide helps users of the original `QDMpy` package (import name `QDMpy`) migrate to
the redesigned `qdmpy-core` package (import name `qdmpy`).

---

## Key Philosophy Changes

| Aspect | Old QDMpy | New qdmpy-core |
|---|---|---|
| **Main object** | Monolithic `QDM` class | Separate `Measurement` + `QDMResult` |
| **Data format** | NumPy arrays | xarray DataArrays with named dims |
| **Processing** | Mutating methods on `ODMR` | Immutable processor pipeline |
| **Results** | Properties on `QDM` | Pure data container `QDMResult` |
| **Saving** | MATLAB .mat export | HDF5 `.qdm` archive + `.npz` |
| **Configuration** | `.ini` file | Pydantic settings with TOML/env |

The architecture is now layered and immutable: loading, processing, fitting, and results are
distinct steps that never mutate in place.

---

## Installation

```bash
# Old
pip install QDMpy

# New
pip install qdmpy-core
```

The import name changes from `QDMpy` to `qdmpy`:

```python
# Old
import QDMpy
from QDMpy.core import QDM, ODMR, Fit

# New
import qdmpy
from qdmpy import Measurement, QDMResult
```

---

## Quick Start Comparison

### Old workflow

```python
import QDMpy
from QDMpy.core import QDM

# Load data
qdm = QDM.from_qdmio("/path/to/data/")

# Process
qdm.odmr.bin_data(2)
qdm.odmr.normalize_data()
qdm.correct_glob_fluorescence(0.2)

# Fit
qdm.fit_odmr()

# Access results
b111 = qdm.b111_remanent          # (H, W) in uT
param = qdm.get_param("contrast")  # (n_pol, n_frange, H, W)

# Save
qdm.export_qdmpy("result.npz")
```

### New workflow

```python
import qdmpy

# Load + process (one-liner)
meas = qdmpy.load("/path/to/data/", bin_factor=2, fluorescence_correction=0.2)

# Or step by step
from qdmpy import Measurement
from qdmpy.odmr.processors import (
    BinningProcessor,
    NormalizationProcessor,
    FluorescenceCorrectionProcessor,
)

meas = Measurement.from_folder("/path/to/data/")
pm = meas.odmr.processor_manager
pm.add_processor(BinningProcessor(bin_factor=2))
pm.add_processor(NormalizationProcessor())                     # normalize BEFORE fluorescence correction
pm.add_processor(FluorescenceCorrectionProcessor(0.2))
meas.odmr.process_data()

# Fit
result = meas.fit_odmr()          # returns QDMResult

# Access results
b111 = result.b111_remanent       # (H, W) in uT

# Save / load (two equivalent forms)
qdmpy.save_qdm(result, "result.qdm")
result.save("result.qdm")                    # equivalent convenience method
result2 = qdmpy.load_qdm("result.qdm")
result2 = QDMResult.load("result.qdm")       # equivalent convenience method
```

---

## API Mapping

### Loading Data

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `QDM.from_qdmio(folder)` | `qdmpy.load(folder)` | One-liner with processing |
| `QDM.from_qdmio(folder)` | `Measurement.from_folder(folder)` | Step-by-step control |
| `QDM.from_matlab(files)` | `MatlabLoader(folder).load()` then `ODMRData.from_loader(loader)` | Lower-level access |
| `ODMR.from_qdmio(folder)` | `ODMRData.from_loader(MatlabLoader(folder))` | ODMR data only |

### Processing ODMR Data

!!! warning "Processor order matters"
    Always add `NormalizationProcessor` **before** `FluorescenceCorrectionProcessor`.
    Normalization with `method="mean"` preserves per-pixel baseline variation that the
    fluorescence correction depends on. Reversing the order produces physically incorrect results.

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `odmr.bin_data(n)` | `BinningProcessor(bin_factor=n)` | Add to processor pipeline |
| `odmr.normalize_data()` | `NormalizationProcessor()` | Default method="mean" |
| `odmr.normalize_data(method="max")` | `NormalizationProcessor(method="max")` | Deprecated; loses baseline info |
| `qdm.correct_glob_fluorescence(f)` | `FluorescenceCorrectionProcessor(f)` | Must run after normalization |
| `odmr.apply_outlier_mask()` | `OutlierProcessor()` | Z-score based; see note below |
| `odmr.reset_data()` | `odmr.reset()` | |
| `odmr.remove_overexposed()` | `HotPixelFilter(threshold_std=5)` in `FieldProcessingPipeline` | Applied post-fit |

**Correct pipeline order:**

```python
from qdmpy.odmr.processors import (
    BinningProcessor,
    NormalizationProcessor,
    FluorescenceCorrectionProcessor,
    OutlierProcessor,
)

pm = meas.odmr.processor_manager
pm.add_processor(BinningProcessor(bin_factor=2))
pm.add_processor(NormalizationProcessor())                     # 1. normalize first
pm.add_processor(FluorescenceCorrectionProcessor(correction_factor=0.2))  # 2. then correct
pm.add_processor(OutlierProcessor(z_score_threshold=0.003))   # 3. then mask outliers
meas.odmr.process_data()
```

!!! note "OutlierProcessor change"
    The new `OutlierProcessor` is a z-score filter (`z_score_threshold` parameter only).
    The old `LocalOutlierFactor` and `IsolationForest` methods are not yet ported.

### Fitting

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `qdm.fit_odmr()` | `result = meas.fit_odmr()` | Returns `QDMResult` |
| `qdm.fit_odmr(refit=True)` | Call `meas.fit_odmr()` again | Always creates new result |
| `qdm.set_model_name("ESR14N")` | `meas.fit_odmr(model_name="ESR14N")` | Or set in `FitManager` |
| `qdm.new_fit(model_name="ESR15N")` | `meas.fit_odmr(model_name="ESR15N")` | |
| `qdm.set_constraints("center", 2.8, 2.9)` | `meas.fit_odmr(constraints={"center": (2.8, 2.9)})` | |
| `qdm.reset_constraints()` | Pass `constraints=None` to `fit_odmr()` | |
| `Fit.guess_model_name()` | `qdmpy.fitting.guess.guess_model(flat_data)` | |

### Accessing Results

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `qdm.get_param("resonance")` | `result.get_parameter("center")` | Name changed: "resonance" -> "center" |
| `qdm.get_param("width")` | `result.get_parameter("width")` | |
| `qdm.get_param("contrast")` | `result.contrasts` | See model-specific note below |
| `qdm.get_param("offset")` | `result.get_parameter("offset")` | |
| `qdm.get_param("chi2")` | `result.chi2` | Direct property |
| `qdm.b111_remanent` | `result.b111_remanent` | (H, W) in uT |
| `qdm.b111_induced` | `result.b111_induced` | (H, W) in uT |
| `qdm.b111` | `result.b111` | xr.Dataset with 'remanent', 'induced' |
| `qdm.delta_resonance` | `result.fit_result.delta_resonance` | xr.DataArray, dims (polarity, y, x) |
| `qdm.data_shape` | `result.scan_dimensions` | (rows, cols) |
| `fit.parameter` | `result.fit_result.parameters` | dict of named arrays |
| `fit.get_param("center")` | `result.get_parameter_map("center")` | 2D spatial map |

### Parameter Names Change

The old API used `"resonance"` for the center frequency; the new API uses `"center"`:

```python
# Old
center_freq = qdm.get_param("resonance")  # shape (n_pol, n_frange, H, W)

# New
center_freq = result.get_parameter("center")          # full array
center_map  = result.get_parameter_map("center")      # 2D map (averaged)
```

### Model-Specific Contrast Parameter Names

For multi-peak models (ESR14N, ESR15N), contrast is split per hyperfine peak and the key
changes. Use the `contrasts` property which handles this automatically:

```python
# ESR14N has: contrast_0, contrast_1, contrast_2
# ESR15N has: contrast_0, contrast_1
# ESRSINGLE has: contrast

# WRONG for ESR14N/ESR15N -- raises ParameterError
result.get_parameter("contrast")

# CORRECT -- works for all models
result.contrasts            # returns contrast_0 (first peak) as a safe default
result.linewidths           # similarly safe for width

# Per-peak access (ESR14N example)
c0 = result.get_parameter("contrast_0")
c1 = result.get_parameter("contrast_1")
c2 = result.get_parameter("contrast_2")
```

### Outlier Detection

The new `OutlierProcessor` uses a z-score threshold rather than scikit-learn algorithms:

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `qdm.detect_outliers(method="LocalOutlierFactor")` | `OutlierProcessor(z_score_threshold=0.003)` | Algorithm changed |
| `qdm.detect_outliers(method="IsolationForest")` | Not yet ported | |
| `StatisticsPercentile(...)` | Not yet ported | |
| `qdm.outliers` | Not a direct property; use processed data NaN mask | |
| `qdm.outlier_pdf` | Not ported | |

### Binning

```python
# Old - mutating in place
qdm.bin_data(4)

# New - processor pipeline
from qdmpy.odmr.processors import BinningProcessor
meas.odmr.processor_manager.add_processor(BinningProcessor(bin_factor=4))
meas.odmr.process_data()
```

### ODMR Data Access

In the table below, `odmr` is `meas.odmr` (the `ODMR` manager) and `odmr_data` is
`meas.odmr.processed_data` (an `ODMRData` object):

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `odmr.data` | `meas.odmr.processed_data.data` | xr.DataArray |
| `odmr['+', '<']` | `meas.odmr.processed_data.data.sel(polarity='pos', freq_range='low')` | Named dims |
| `odmr.frequencies` | `meas.odmr.processed_data.frequencies` | (n_frange, n_freq) in GHz |
| `odmr.f_ghz` | `meas.odmr.processed_data.frequencies` | Already GHz |
| `odmr.f_hz` | `meas.odmr.processed_data.frequencies * 1e9` | Convert as needed |
| `odmr.n_pixel` | `meas.odmr.processed_data.data.sizes['y'] * meas.odmr.processed_data.data.sizes['x']` | |
| `odmr.n_pol` | `meas.odmr.processed_data.data.sizes['polarity']` | |
| `odmr.n_frange` | `meas.odmr.processed_data.data.sizes['freq_range']` | |
| `odmr.spectrum(y, x)` | `meas.odmr.spectrum(y, x, polarity='neg', freq_range='low')` | Named args |
| `odmr.plot_spectra(y, x)` | `meas.odmr.plot_spectra(y, x)` | Same |
| `odmr.check_glob_fluorescence(gf)` | `qdmpy.plotting.plot_fluorescence_correction(odmr_data, gf)` | |
| `odmr.get_most_divergent_from_mean()` | Not ported | |

### ESR Models

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `from QDMpy.core import ESR14N, ESR15N, ESRSINGLE` | `from qdmpy.fitting.models import ESR14N, ESR15N, ESRSINGLE` | |
| `IMPLEMENTED` dict | `ModelRegistry.all()` | |
| `PEAK_TO_TYPE` dict | `ModelRegistry` | |
| `esr14n(x, params)` | `from qdmpy.fitting.models import esr14n` | Same function sig |
| `esr15n(x, params)` | `from qdmpy.fitting.models import esr15n` | Same |
| `esrsingle(x, params)` | `from qdmpy.fitting.models import esrsingle` | Same |

### I/O and Persistence

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `qdm.export_qdmpy(path)` | `qdmpy.save_npz(result, path)` or `qdmpy.save_qdm(result, path)` | NPZ is fit-only, `.qdm` is full archive |
| `qdm.export_qdmio(path)` | **Not ported** | MATLAB export; see workaround below |
| `qdm.export_MMT(path)` | **Not ported** | Multi-binning test format not available |
| `get_image(folder, files)` | `qdmpy.io.get_image(folder, files)` | Same signature |
| `get_image_file(files)` | `qdmpy.io.get_image_file(folder, files)` | Folder arg added |
| `has_csv(files)` | `qdmpy.io.has_csv(files)` | Same |
| (no equivalent) | `qdmpy.load_qdm(path)` or `qdmpy.load_npz(path)` | New HDF5 format + safe NPZ loader |
| (no equivalent) | `result.save(path)` / `QDMResult.load(path)` | Object-level convenience |

#### NPZ safety and legacy migration

`qdmpy.save_npz()` writes a pickle-free format (JSON metadata + numeric arrays),
and `qdmpy.load_npz()` uses safe loading (`allow_pickle=False`) for new files.

Older NPZ files created by early qdmpy-core versions used pickled object arrays.
These legacy files are still accepted for one migration release, with a
deprecation warning. Migrate them by loading once and re-saving:

```python
legacy = qdmpy.load_npz('legacy_result.npz')
qdmpy.save_npz(legacy, 'legacy_result_migrated.npz')
```

For long-term archives and full reproducibility (images, optional Bxyz, field
sources), prefer the `.qdm` format.

### Configuration

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `QDMpy.make_configfile()` | `qdmpy.settings.make_configfile()` | |
| `QDMpy.load_config()` | `qdmpy.get_settings()` | Returns `QDMpySettings` |
| `QDMpy.reset_config()` | `qdmpy.reset_settings()` | Name changed |
| `QDMpy.SETTINGS["fit"]["estimator"]` | `qdmpy.get_settings().fit.estimator` | Pydantic model |
| `PYGPUFIT_PRESENT` flag | `qdmpy.is_pygpufit_available()` | Function call |

### Utilities

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `from QDMpy.utils import double_norm` | `from qdmpy.utils import double_norm` | Same |
| `from QDMpy.utils import millify` | `from qdmpy.utils import millify` | Same |
| `from QDMpy.utils import rms` | `from qdmpy.utils import rms` | Same |
| `from QDMpy.utils import idx2rc, rc2idx` | `from qdmpy.utils import idx2rc, rc2idx` | Same |
| `from QDMpy.utils import polyfit2d` | `from qdmpy.utils import polyfit2d` | Same |
| `qdm.rc2idx(rc)` | `qdmpy.utils.rc2idx(rc, result.scan_dimensions)` | Moved to standalone |
| `qdm.idx2rc(idx)` | `qdmpy.utils.idx2rc(idx, result.scan_dimensions)` | Moved to standalone |

### Exceptions

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `CantImportError` | `qdmpy.exceptions.DependencyError` | Renamed |
| `WrongFileNumber` | `qdmpy.exceptions.DataLoadError` | Generalised |

### Plotting

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `check_fit_pixel(qdm, idx)` | `meas.display(result)` then click pixel | Interactive |
| `plot_fit_params(ax, qdm)` | `qdmpy.plotting.plot_fit_result_overview(result)` | |
| `plot_light_img(ax, data, img)` | `qdmpy.plotting.plot_measurement_images(meas)` | |
| `plot_fluorescence(ax, data)` | included in `plot_measurement_images` | |
| (manual spectrum plot) | `qdmpy.plotting.plot_odmr_spectra(odmr_data, y, x)` | |

### Testing / Synthetic Data

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `make_dummy_data(model, n_freqs, dims)` | `qdmpy.make_synthetic_odmr_data(shape, n_freq, model_name)` | Always 2-pol 2-frange |
| `write_test_qdmio_file(path)` | Not ported as public API | Use `make_synthetic_odmr_data` |

### Constants

!!! warning "GAMMA_NV unit change: factor of 10^6"
    The gyromagnetic ratio changed units. Code that used `GAMMA` directly must be updated:

    ```python
    # Old: GAMMA = 28.024 / 1e6  (GHz/uT) -- divide delta_GHz by GAMMA to get uT
    b_ut = delta_ghz / GAMMA

    # New: GAMMA_NV = 28.024  (GHz/T) -- divide delta_GHz by GAMMA_NV to get T, then convert
    from qdmpy.constants import GAMMA_NV
    b_ut = delta_ghz / GAMMA_NV * 1e6   # GHz / (GHz/T) * (1e6 uT/T) = uT
    ```

| Old QDMpy | New qdmpy-core | Notes |
|---|---|---|
| `GAMMA = 28.024 / 1e6` (GHz/uT) | `qdmpy.constants.GAMMA_NV = 28.024` (GHz/T) | **Unit change** |
| - | `qdmpy.constants.D_ZFS = 2.870` | New |
| - | `qdmpy.constants.AHYP_14N = 0.002158` | New |
| - | `qdmpy.constants.AHYP_15N = 0.0015` | New |

---

## New Features in qdmpy-core

These features did not exist in the old QDMpy and are entirely new:

### Magnetic Field Reconstruction (Bxyz)

Reconstruct the full 3D magnetic field from B111 measurements using Fourier inversion:

```python
magnetic_map = result.magnetic_map  # lazily computed on first access
bz = magnetic_map.bz                # xr.DataArray in uT
qdmpy.plotting.plot_magnetic_component(magnetic_map, component="bz")
```

### NV Axis Configuration

The NV axis used for Bxyz reconstruction is read from global settings by default.
To override it, set `nv_axis` directly on `QDMResult` before accessing `magnetic_map`:

```python
result = meas.fit_odmr()
result.nv_axis = (0.0, 0.8165, 0.5774)   # custom NV axis unit vector
magnetic_map = result.magnetic_map         # uses the custom axis
```

### Field Source Fitting

Fit magnetic dipole sources to the measured field using `pypole`:

```python
from qdmpy import MagneticSource, MagneticModel, fit_sources

source = MagneticSource(
    center=(100, 200),         # (x, y) in pixels
    half_extent=(20, 20),      # (dx, dy) ROI half-widths in pixels
    pixel_spacing=4e-6,        # meters per pixel
    model=MagneticModel(inclination=90, declination=0, magnetic_moment=1e-12),
)
result.field_sources = [source]

# standoff_m is the sensor-to-sample distance in metres
fitted = fit_sources(result, standoff_m=5e-6)
for r in fitted:
    print(r.source.model)     # updated MagneticModel with fitted parameters
```

### Field Map Post-Processing

Apply corrections to field maps after fitting:

```python
from qdmpy.field_processing import (
    FieldProcessingPipeline,
    BlankSubtractor,
    HotPixelFilter,
    QuadraticBackgroundSubtractor,
    UpwardContinuation,
)

pipeline = (
    FieldProcessingPipeline()
    .add(HotPixelFilter(threshold_std=5))
    .add(QuadraticBackgroundSubtractor(polyfit_order=3))
    .add(UpwardContinuation(height_m=50e-6))
)
corrected_bz = pipeline.process(result.magnetic_map.bz)
```

### Spectral Folding

Fold ODMR spectra about the per-pixel D_ZFS for improved sensitivity:

```python
folded = meas.fold_odmr()
result = meas.fit_folded_odmr(folded)
qdmpy.plotting.plot_folding_overview(folded)
```

### HDF5 Archive Format (.qdm)

The new `.qdm` format saves all data including images and field sources:

```python
qdmpy.save_qdm(result, "measurement.qdm", include_bxyz=True)
result.save("measurement.qdm")              # equivalent convenience method
result = qdmpy.load_qdm("measurement.qdm")
result = QDMResult.load("measurement.qdm")  # equivalent convenience method
```

### Rich Exception Hierarchy

Granular exceptions replace generic Python errors:

```python
from qdmpy.exceptions import (
    FitNotPerformedError,
    ModelNotFoundError,
    DataLoadError,
    FoldingError,
)
```

---

## Features Not Yet Migrated

The following features from old QDMpy are not yet available in qdmpy-core:

| Feature | Old API | Status | Notes |
|---|---|---|---|
| MATLAB export | `qdm.export_qdmio()` | Not planned | See workaround below |
| Multi-binning test export | `qdm.export_MMT()` | Not planned | Dev utility; script manually if needed |
| LocalOutlierFactor outlier detection | `detect_outliers(method="LocalOutlierFactor")` | Not yet ported | Use `OutlierProcessor(z_score_threshold=...)` |
| IsolationForest outlier detection | `detect_outliers(method="IsolationForest")` | Not yet ported | Use `OutlierProcessor(z_score_threshold=...)` |
| StatisticsPercentile outlier detection | `StatisticsPercentile(...)` | Not yet ported | |
| Outlier DataFrame | `qdm.outlier_pdf` | Not ported | NaN mask in processed data is equivalent |
| Most-divergent pixel | `odmr.get_most_divergent_from_mean()` | Not ported | Compute manually from mean spectrum |
| Binned pixel index lookup | `odmr.get_binned_pixel_indices(x, y)` | Not ported | Not needed with xarray named coords |

### Workaround: MATLAB-Compatible Export

If you need a MATLAB-compatible `.mat` file:

```python
import scipy.io

scipy.io.savemat("B111dataToPlot.mat", {
    "b111Remanent": result.b111_remanent,
    "b111Induced": result.b111_induced,
    "pixelSize": result.pixel_spacing,
})
```

---

## Data Shape Conventions

Both old and new use the same fundamental convention, but array structures differ:

| Quantity | Old shape | New shape | Notes |
|---|---|---|---|
| ODMR data | `(n_pol, n_frange, n_pixel, n_freq)` | `(n_pol, n_frange, H, W, n_freq)` | New keeps spatial dims separate |
| B111 maps | `(H, W)` | `(H, W)` | Same |
| Fit parameters (raw) | `(n_pol, n_frange, H*W, n_params)` | `(n_pol, n_frange, H, W, n_params)` | New uses 5D |
| Fit parameter map | `result.get_param(p)` -> `(n_pol, n_frange, H, W)` | `result.get_parameter_map(p)` -> `(H, W)` | New averages by default |
| delta_resonance | `(n_pol, 2, H, W)` | xr.DataArray with dims `(polarity, y, x)` | Signed values; use `.sel(polarity=...)` |

---

## Minimum Working Example

```python
import qdmpy

# 1. Load and process
meas = qdmpy.load(
    "/path/to/data/",
    bin_factor=2,
    fluorescence_correction=0.2,
    model="ESR14N",
)

# 2. Fit
result = meas.fit_odmr()

# 3. Inspect
print(result.scan_dimensions)           # (H, W)
print(result.get_fit_quality_metrics())

# 4. Maps
b111 = result.b111_remanent             # (H, W) uT
meas.display(result)                    # comprehensive overview

# 5. Save
result.save("output.qdm")
```
