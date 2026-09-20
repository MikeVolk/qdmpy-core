# QEP-043 — `QDMpy.load()` Entry-Point Function

**Status:** Implemented (2026-02-21)
**Created:** 2026-02-21

---

## Motivation

The current minimum path to a fit result requires 10+ lines across 5 import
paths. A scientist who just received a data folder cannot start without reading
source code or finding an example notebook:

```python
from QDMpy.odmr.io import MatlabLoader
from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.manager import ODMR
from QDMpy.odmr.processors import BinningProcessor, NormalizationProcessor
from QDMpy.measurement import Measurement

loader = MatlabLoader("/data/FOV18x")
odmr_data = ODMRData.from_loader(loader)
odmr = ODMR(odmr_data)
odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
odmr.processor_manager.add_processor(NormalizationProcessor())
odmr.process_data()
light_img, laser_img = ???   # not documented
measurement = Measurement(odmr=odmr, light_image=light_img,
                          laser_image=laser_img, pixel_spacing=4e-6)
result = measurement.fit_odmr()
```

Compare with numpy: `np.load(path)`. Or xarray: `xr.open_dataset(path)`.
Scientific Python users expect a top-level loader.

---

## Goals

1. `QDMpy.load(path)` creates a fully-configured `Measurement` ready to fit.
2. Optional keyword arguments expose the most common customisations without
   requiring knowledge of the full API.
3. Advanced users can inspect and modify the returned `Measurement` before
   fitting.
4. The function handles image loading internally (light + laser images from
   standard MATLAB data folder layout).

---

## Design

### 5.1  Function signature

```python
def load(
    path: str | PathLike,
    *,
    bin_factor: int = 1,
    model: str = 'auto',
    pixel_spacing: float = 4e-6,
    normalize: bool = True,
    output_directory: str | PathLike | None = None,
) -> Measurement:
    """Load ODMR data from a folder and return a ready-to-fit Measurement.

    Args:
        path: Folder containing MATLAB .mat files from the QDM microscope.
        bin_factor: Spatial binning factor (1 = no binning, 2 = 2×2 bins, …).
        model: ESR model name ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE').
        pixel_spacing: Physical pixel size in metres (default 4 µm).
        normalize: Apply max-normalisation to ODMR spectra (default True).
        output_directory: Directory for saved outputs. Defaults to path/results.

    Returns:
        Measurement configured and ready for fit_odmr().

    Example:
        >>> import QDMpy
        >>> result = QDMpy.load('/data/FOV18x').fit_odmr()
        >>> result.b111_remanent
    """
```

### 5.2  Implementation sketch

```python
def load(path, *, bin_factor=1, model='auto', pixel_spacing=4e-6,
         fluorescence=0.2, # note I added this the name could be better
         normalize=True, output_directory=None):
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.manager import ODMR
    from QDMpy.odmr.processors import BinningProcessor, NormalizationProcessor
    from QDMpy.measurement import Measurement
    from QDMpy.io import get_image

    path = Path(path)
    loader = MatlabLoader(path)
    odmr = ODMR(ODMRData.from_loader(loader))

    if bin_factor > 1:
        odmr.processor_manager.add_processor(BinningProcessor(bin_factor))
    if normalize:
        odmr.processor_manager.add_processor(NormalizationProcessor())
    if fluorescence:
        odmr.processor_manager.add_processor(FluorescenceProcessor(fluorescence)) # note I added this may be incorrect
    odmr.process_data()

    # note what happens if they dont exist
    light_image = get_image(path, kind='light')
    laser_image = get_image(path, kind='laser')

    return Measurement(
        odmr=odmr,
        light_image=light_image,
        laser_image=laser_image,
        pixel_spacing=pixel_spacing,
        fit_model=model,
        output_directory=output_directory or path / 'results',
    )
```

### 5.3  Minimal user workflow after this QEP

```python
import QDMpy

result = QDMpy.load('/data/FOV18x').fit_odmr()
print(result.b111_remanent)
```

Or with customisation:

```python
measurement = QDMpy.load('/data/FOV18x', bin_factor=4, model='ESR14N')
# inspect or modify measurement before fitting
result = measurement.fit_odmr(constraints={'width': {'vmax': 0.05}})
```

### 5.4  Location

`load()` lives in `src/QDMpy/__init__.py` (or a thin `src/QDMpy/_load.py`
imported there). It is exported in `__all__`.

---

## Interaction with QEP-042

QEP-042 (top-level exports) is a prerequisite — `load()` needs `Measurement`,
`ODMR`, etc. to be importable but the function itself consolidates them so the
user need not import anything individually.

---

## Alternatives Considered

### A. `Measurement.from_folder(path, **kwargs)` classmethod
Viable. Slightly less discoverable than `QDMpy.load()` (user must know the
`Measurement` class first). Both can coexist; `QDMpy.load` delegates to
`Measurement.from_folder` internally.

### B. CLI only (`qdmpy fit /data/FOV18x`)
Complements but does not replace — interactive notebook users need a Python API.

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/__init__.py` | Add `load()` function; add to `__all__` |
| `src/QDMpy/measurement.py` | Add `Measurement.from_folder()` classmethod (delegated to by `load()`) |
| `tests/test_load.py` | **New** — unit tests with mocked loader; integration test with real folder |
