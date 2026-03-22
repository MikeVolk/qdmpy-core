# QEP-034 — Field Processing Pipeline and B111 → Bxyz Reconstruction

**Status:** Implemented
**Created:** 2026-02-21
**Supersedes:** QEP-027 (merged here; `B111ToBzConverter` dropped)
**Depends on:** QEP-025 (semantic coordinates), QEP-029 (stateless FitResult)

---

## Motivation

After fitting, a standard post-processing chain transforms B111 maps before quantitative
analysis. QDMpy has no equivalent. The MATLAB reference implements five steps:
`filter_hot_pixels`, `QuadBGsub`, `UpCont`, `subtract_blank`, and `B111ToBz_CommLine`.

Beyond preprocessing, the central derived quantity for most analyses is not B111 (the NV
projection) but the full 3D vector field **Bxyz**. QDMlab reconstructs it in two steps:

1. **B111 → Bz** (`QDMBzFromBu.m`) — invert the NV projection geometry in Fourier space
2. **Bz → Bx, By** (`MITBxByFromBz.m`) — apply free-space Maxwell relations

Since Bz is the intermediate of step 1, combining the two steps yields Bx, By, Bz, and
Btotal from a single FFT2 + three IFFT2s, with no redundant computation. A separate
`B111ToBzConverter` (as sketched in QEP-027) is therefore unnecessary.

---

## Goals

1. `BaseFieldProcessor` + `FieldProcessingPipeline` — composable preprocessing chain
   operating on `xr.DataArray` (pixel_spacing carried in `.attrs`).
2. Concrete processors: `HotPixelFilter`, `QuadraticBackgroundSubtractor`,
   `UpwardContinuation`, `BlankSubtractor`.
3. `MagneticMap` — result object holding `{b111, bx, by, bz, btotal}` as `xr.DataArray`,
   with `display()` and `to_dataset()`. Constructed via `MagneticMap.from_b111()`.
4. `NvSettings` added to `QDMpySettings` — NV axis and related geometry, usable without
   a `Measurement` object.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Physics

### Free-space field relations

Above the source plane (no currents), `∇ × B = 0` and `∇ · B = 0`. In 2D Fourier space,
where fields decay as `exp(-k·z)` with `k = ||(kx, ky)||`:

```
B̃x = -i·kx/k · B̃z
B̃y = -i·ky/k · B̃z
```

### B111 → Bz

With NV unit vector `û = (ux, uy, uz)`:

```
B111 = û · B  →  B̃111 = B̃z · (uz·k - ux·i·kx - uy·i·ky) / k
```

Inverting:

```
B̃z = B̃111 · k / (uz·k - ux·i·kx - uy·i·ky)
```

The denominator is zero at `k = 0` (DC) and on a line through the origin in k-space.
Both are regularised by adding a small epsilon to all wavenumber components.

### Combined pipeline

```
FFT2(B111)
    ×  k / (uz·k - ux·i·kx - uy·i·ky)  →  F̃z        [B111 → Bz]
    ×  -i·kx/k                           →  F̃x        [Bz → Bx]
    ×  -i·ky/k                           →  F̃y        [Bz → By]
IFFT2(F̃x), IFFT2(F̃y), IFFT2(F̃z)
```

One FFT2 + three IFFT2s total.

### Default NV axis

QDM2 standard `[111]` orientation with `[1-10]` along x:

```
û_default = (0, √(2/3), 1/√3)  ≈  (0.0, 0.8165, 0.5774)
```

Matches the hardcoded `U` vector in `B111ToBz_CommLine.m`. Configurable via `NvSettings`.

---

## Design

### 4.1  `NvSettings` (new Settings submodel)

Added to `QDMpySettings` under the key `nv`:

```python
class NvSettings(BaseModel):
    """NV centre geometry for B111 → Bxyz reconstruction."""

    axis: tuple[float, float, float] = Field(
        default=(0.0, 0.8164966, 0.5773503),
        description='NV unit vector (ux, uy, uz) in lab frame. '
                    'Default: QDM2 [111] orientation.',
    )
    epsilon: float = Field(
        default=1e-30,
        description='Regularisation term added to wavenumbers to avoid k=0 singularity.',
    )

    model_config = ConfigDict(extra='ignore')
```

```python
class QDMpySettings(BaseSettings):
    ...
    nv: NvSettings = Field(default_factory=NvSettings, description='NV geometry settings')
```

TOML override example:
```toml
[nv]
axis = [0.0, 0.8164966, 0.5773503]
```

---

### 4.2  `BaseFieldProcessor`

```python
from abc import abstractmethod
from pydantic import BaseModel, ConfigDict
import xarray as xr

class BaseFieldProcessor(BaseModel):
    """Abstract base for all field-map processors.

    Processors are Pydantic frozen models: all configuration lives in fields
    set at construction; ``process()`` is a pure function of its argument.
    ``pixel_spacing`` (in metres) must be present in ``field_map.attrs``.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Transform a (H, W) field map.

        Args:
            field_map: DataArray with dims (y, x), values in µT,
                       and ``pixel_spacing`` (float, metres) in ``.attrs``.

        Returns:
            Processed DataArray with identical dims, coords, and attrs.
            Input is never mutated.
        """

    @staticmethod
    def _pixel_spacing(field_map: xr.DataArray) -> float:
        if 'pixel_spacing' not in field_map.attrs:
            raise ValueError("field_map.attrs must contain 'pixel_spacing' (metres)")
        return float(field_map.attrs['pixel_spacing'])
```

---

### 4.3  `HotPixelFilter`

Matches `filter_hot_pixels` in QDMlab.

```python
from typing import Literal

class HotPixelFilter(BaseFieldProcessor):
    threshold_sigma: float = 5.0
    window_size: int = 3                              # half-width; full window (2w+1)²
    replacement: Literal['mean', 'nan', 'zero'] = 'mean'
    absolute_threshold: float | None = None           # always filter |field| > this first
```

**Algorithm:**

```
detection = field_map.values

1. Pre-filter: mark |detection| > absolute_threshold as potential outliers (if set)
2. Compute nanmedian and nanstd of detection
3. Outlier mask: |detection - nanmedian| > threshold_sigma * nanstd  (union with step 1)
4. For each outlier pixel:
     neighbours = data[r-w:r+w+1, c-w:c+w+1]  (clipped at image boundary)
     apply replacement: nanmean(neighbours excl. centre) | NaN | 0.0
5. Return new DataArray (same coords/attrs) with outliers replaced
```

---

### 4.4  `QuadraticBackgroundSubtractor`

Matches `QuadBGsub.m`.

```python
class QuadraticBackgroundSubtractor(BaseFieldProcessor):
    degree: int = 2                      # 1 = plane, 2 = quadratic surface
    mask: tuple[tuple[int, ...], ...] | None = None
    # Pixel-index mask of rows to EXCLUDE from fit (e.g. over the sample).
    # Stored as nested tuples for Pydantic hashability; converted to NDArray internally.
```

**Algorithm:**

```
Build (H*W, n_terms) design matrix with polynomial features up to degree:
    degree=1: [1, x, y]
    degree=2: [1, x, y, x², xy, y²]
    Normalise x, y to [-1, 1] for numerical stability.

Fit only ~mask pixels: coeffs = lstsq(A[active], data[active], rcond=None)
Surface = A @ coeffs   (evaluated at ALL pixels)
Return field - surface  (as new DataArray, attrs preserved)
```

---

### 4.5  `UpwardContinuation`

Matches `UpCont.m`.

```python
class UpwardContinuation(BaseFieldProcessor):
    dz: float              # continuation height in metres (>0 = away from source)
    padding_factor: float = 3.0
    oversampling: int = 2
```

**Algorithm:**

```python
def process(self, field_map: xr.DataArray) -> xr.DataArray:
    ps = self._pixel_spacing(field_map)
    data = field_map.values
    ny, nx = data.shape

    # Embed in padded array
    py, px = int(ny * self.padding_factor), int(nx * self.padding_factor)
    padded = np.zeros((py, px))
    oy, ox = (py - ny) // 2, (px - nx) // 2
    padded[oy:oy+ny, ox:ox+nx] = data

    # Wavenumber grid
    ny_f, nx_f = py * self.oversampling, px * self.oversampling
    fy = np.fft.fftfreq(ny_f, d=ps)
    fx = np.fft.fftfreq(nx_f, d=ps)
    Fx, Fy = np.meshgrid(fx, fy)
    k = 2 * np.pi * np.sqrt(Fx**2 + Fy**2)

    # Apply continuation filter and crop back
    H = np.exp(-self.dz * k)
    out = np.real(np.fft.ifft2(np.fft.fft2(padded, s=(ny_f, nx_f)) * H))
    result = out[:py, :px][oy:oy+ny, ox:ox+nx]

    return xr.DataArray(result, dims=field_map.dims,
                        coords=field_map.coords, attrs=field_map.attrs)
```

`dz < 0` (downward continuation) amplifies noise — log a warning but proceed.

---

### 4.6  `BlankSubtractor`

```python
class BlankSubtractor(BaseFieldProcessor):
    blank: tuple[tuple[float, ...], ...]
    # Pre-measured background map stored as nested tuples (Pydantic-serialisable).
    # Must match the field map shape.
```

```python
def process(self, field_map: xr.DataArray) -> xr.DataArray:
    blank = np.array(self.blank)
    if blank.shape != field_map.shape:
        raise DataShapeError(
            f'Blank shape {blank.shape} != field shape {tuple(field_map.shape)}'
        )
    return field_map - blank   # xarray preserves coords/attrs
```

Spatial alignment of the blank (if required) is the caller's responsibility.

---

### 4.7  `FieldProcessingPipeline`

```python
class FieldProcessingPipeline:
    """Sequential chain of BaseFieldProcessor steps operating on xr.DataArray."""

    def __init__(self) -> None:
        self._processors: list[BaseFieldProcessor] = []

    def add(self, processor: BaseFieldProcessor) -> FieldProcessingPipeline:
        """Append a processor. Returns self for method chaining."""
        self._processors.append(processor)
        return self

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Apply all processors in order. Input is never mutated."""
        result = field_map.copy()
        for proc in self._processors:
            result = proc.process(result)
            logger.debug(
                'Field processor applied',
                processor=proc.__class__.__name__,
                shape=result.shape,
            )
        return result
```

**Attaching pixel_spacing before processing:**

```python
b111 = fit_result.b111['remanent'].assign_attrs(pixel_spacing=fit_result.pixel_spacing)
preprocessed = pipeline.process(b111)
```

This is the caller's responsibility — no hidden coupling between `FieldProcessingPipeline`
and `FitResult`.

---

### 4.8  `MagneticMap`

Pure result object. Reconstruction math lives in `from_b111()`.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import xarray as xr
import numpy as np

@dataclass(frozen=True)
class MagneticMap:
    """Full 3D magnetic field reconstructed from a B111 map.

    All field components are xr.DataArray with dims (y, x) and units µT.
    """

    b111: xr.DataArray
    bx: xr.DataArray
    by: xr.DataArray
    bz: xr.DataArray
    btotal: xr.DataArray
    nv_axis: tuple[float, float, float]

    @classmethod
    def from_b111(
        cls,
        b111: xr.DataArray,
        nv_axis: tuple[float, float, float] | None = None,
        epsilon: float | None = None,
    ) -> MagneticMap:
        """Reconstruct Bxyz from a preprocessed B111 map.

        Args:
            b111: DataArray with dims (y, x), values in µT, and
                  ``pixel_spacing`` (metres) in ``.attrs``.
            nv_axis: NV unit vector (ux, uy, uz). Defaults to
                     ``get_settings().nv.axis``.
            epsilon: k=0 regularisation. Defaults to
                     ``get_settings().nv.epsilon``.

        Returns:
            MagneticMap with b111, bx, by, bz, btotal.
        """
        from QDMpy.settings import get_settings
        settings = get_settings()
        nv = nv_axis or settings.nv.axis
        eps = epsilon if epsilon is not None else settings.nv.epsilon
        ps = float(b111.attrs['pixel_spacing'])

        bx_arr, by_arr, bz_arr = _reconstruct_bxyz(b111.values, ps, nv, eps)
        btotal_arr = np.sqrt(bx_arr**2 + by_arr**2 + bz_arr**2)

        def _da(arr: np.ndarray, name: str) -> xr.DataArray:
            return xr.DataArray(
                arr, dims=b111.dims, coords=b111.coords,
                attrs={**b111.attrs, 'component': name},
            )

        return cls(
            b111=b111,
            bx=_da(bx_arr, 'Bx'),
            by=_da(by_arr, 'By'),
            bz=_da(bz_arr, 'Bz'),
            btotal=_da(btotal_arr, 'Btotal'),
            nv_axis=nv,
        )

    def to_dataset(self) -> xr.Dataset:
        """Return all components as a single xr.Dataset."""
        return xr.Dataset(
            {'b111': self.b111, 'Bx': self.bx, 'By': self.by,
             'Bz': self.bz, 'Btotal': self.btotal},
            attrs={'units': 'µT', 'nv_axis': list(self.nv_axis)},
        )

    def display(
        self,
        component: Literal['b111', 'Bx', 'By', 'Bz', 'Btotal'] = 'Bz',
        **imshow_kwargs: object,
    ) -> None:
        """Quick matplotlib display of one component."""
        import matplotlib.pyplot as plt
        da = getattr(self, component.lower() if component == 'b111' else component)
        da.plot(**imshow_kwargs)
        plt.title(component)
        plt.show()

    def save(self, path: str | Path) -> None:
        """Save all components to NetCDF."""
        self.to_dataset().to_netcdf(path)
        logger.info('MagneticMap saved', path=str(path))
```

**Core reconstruction function** (module-private):

```python
def _reconstruct_bxyz(
    b111: np.ndarray,
    pixel_spacing: float,
    nv_axis: tuple[float, float, float],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct (Bx, By, Bz) from B111 in the Fourier domain.

    References:
        QDMBzFromBu.m (Eduardo A. Lima, 2017)
        MITBxByFromBz.m (Eduardo A. Lima, 2007)
    """
    ux, uy, uz = nv_axis
    ny, nx = b111.shape

    fy = np.fft.fftfreq(ny, d=pixel_spacing) + epsilon
    fx = np.fft.fftfreq(nx, d=pixel_spacing) + epsilon
    Fx, Fy = np.meshgrid(fx, fy)
    kx = 2 * np.pi * Fx
    ky = 2 * np.pi * Fy
    k = np.sqrt(kx**2 + ky**2)

    F_b111 = np.fft.fft2(b111)

    # Step 1: B111 → Bz
    H_bz = k / (uz * k - uy * 1j * ky - ux * 1j * kx)
    F_bz = F_b111 * H_bz

    # Step 2: Bz → Bx, By  (free-space Maxwell, Egli & Heller eq. 12)
    bz = np.real(np.fft.ifft2(F_bz))
    bx = np.real(np.fft.ifft2(F_bz * (-1j * kx / k)))
    by = np.real(np.fft.ifft2(F_bz * (-1j * ky / k)))

    return bx, by, bz
```

---

## Typical Usage

```python
from QDMpy.field_processing import (
    FieldProcessingPipeline,
    HotPixelFilter,
    QuadraticBackgroundSubtractor,
    UpwardContinuation,
)
from QDMpy.magnetic_map import MagneticMap

# Preprocessing pipeline
pipeline = (
    FieldProcessingPipeline()
    .add(HotPixelFilter(threshold_sigma=5.0))
    .add(QuadraticBackgroundSubtractor(degree=2))
    .add(UpwardContinuation(dz=5e-6))
)

b111 = fit_result.b111['remanent'].assign_attrs(
    pixel_spacing=fit_result.pixel_spacing
)
preprocessed = pipeline.process(b111)

# Reconstruct full 3D field (NV axis from settings by default)
mag = MagneticMap.from_b111(preprocessed)

mag.bz                        # xr.DataArray (H, W), µT
mag.display('Bz')
mag.save('/data/FOV1/magnetic_map.nc')

# Override NV axis for non-standard orientation
import numpy as np
phi, theta = np.radians(45), np.radians(54.7356)
nv = (np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))
mag = MagneticMap.from_b111(preprocessed, nv_axis=nv)
```

---

## New Modules

```
src/QDMpy/
├── field_processing.py    # BaseFieldProcessor, FieldProcessingPipeline, concrete processors
└── magnetic_map.py        # MagneticMap, _reconstruct_bxyz
```

`NvSettings` is added to `settings.py`. No other existing modules are modified.

---

## Changes vs QEP-027

| QEP-027 | QEP-034 |
|---------|---------|
| Processors take `(NDArray, pixel_spacing)` | Processors take `xr.DataArray` (pixel_spacing in `.attrs`) |
| `B111ToBzConverter` (B111 → Bz only) | Dropped — Bz falls out free from `MagneticMap.from_b111()` |
| No result object | `MagneticMap` with `{b111, bx, by, bz, btotal, display, save}` |
| NV axis on the converter instance | `NvSettings` in `QDMpySettings` (overridable per call) |
| Depends on QEP-026 ResultStack | No ResultStack dependency |

---

## Alternatives Considered

### A. `to_magnetic_map()` method on `FitResult`
Rejected. `FitResult` owns fitting outputs; field reconstruction is post-processing.
Keeping layers clean means `FitResult` never imports `MagneticMap`. Callers do
`MagneticMap.from_b111(fit_result.b111['remanent'].assign_attrs(...))` explicitly.

### B. `B111ToBxyzConverter` as a `BaseFieldProcessor` pipeline step
Rejected. Its output type (`MagneticMap`) is incompatible with the pipeline contract
(`xr.DataArray → xr.DataArray`). Making it a pipeline step would require a union return
type and special-casing in `FieldProcessingPipeline.process()`. Keeping reconstruction
outside the pipeline as `MagneticMap.from_b111()` preserves uniformity.

### C. Multi-family Bxyz (four NV orientations simultaneously)
Deferred. Requires four B111 maps and simultaneous inversion — a separate QEP.
Single-family reconstruction covers the standard `[111]` diamond case.

### D. Thin `FieldMap` wrapper instead of `xr.DataArray`
Deferred. `xr.DataArray` with `pixel_spacing` in `.attrs` is sufficient now. A wrapper
can be introduced later without changing the processor interface.
