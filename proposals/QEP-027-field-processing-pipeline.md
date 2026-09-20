# QEP-027 — Field Processing Pipeline

**Status:** Superseded by QEP-034 (implemented)
**Created:** 2026-02-18
**Depends on:** QEP-026 (ResultStack)

---

## Motivation

After fitting, QDMlab applies a standard chain of post-processing steps to B111/Bz maps
before any quantitative analysis:

| Step | QDMlab function | Purpose |
|------|----------------|---------|
| Hot pixel removal | `filter_hot_pixels` | Replace outliers with local mean |
| Quadratic background subtraction | `QuadBGsub` | Remove large-scale field drift |
| Upward continuation | `UpCont` | Project field to different height |
| Blank subtraction | `subtract_blank` | Remove diamond/setup contribution |
| B111 → Bz conversion | `B111ToBz_CommLine` | Convert diagonal to vertical component |

QDMpy has no equivalent of any of these. They are currently implemented on raw 2D numpy
arrays in MATLAB with no abstraction. This QEP introduces a composable
`FieldProcessingPipeline` that mirrors the `ODMRProcessorManager` pattern already in the
codebase.

---

## Goals

1. `BaseFieldProcessor` — ABC with a uniform `process(map, pixel_spacing) → map` interface.
2. Concrete processors: `HotPixelFilter`, `QuadraticBackgroundSubtractor`,
   `UpwardContinuation`, `BlankSubtractor`, `B111ToBzConverter`.
3. `FieldProcessingPipeline` — sequential chain with `ResultStack` integration.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### 3.1  `BaseFieldProcessor`

`BaseFieldProcessor` follows the pattern established by QEP-030 (`BaseProcessor`):

```python
from pydantic import BaseModel, ConfigDict
from abc import abstractmethod

class BaseFieldProcessor(BaseModel):
    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def process(self, field_map: NDArray, pixel_spacing: float) -> NDArray:
        """Transform a (H, W) field map in-place-safe fashion.

        Args:
            field_map: 2D array of field values (µT or Gauss, consistent throughout).
            pixel_spacing: Physical distance between pixels in metres.

        Returns:
            Processed (H, W) field map. Input is never mutated.
        """
```

**Architecture note:** Unlike ODMR processors (QEP-030), field processors do NOT store
data (e.g., `chi2_map`, `blank_map`) as Pydantic fields. See sections 3.2 and 3.5 for
how stateless design is preserved. `pixel_spacing` is passed explicitly at process time
rather than stored at construction, making processors stateless and reusable across data
with different resolutions. Pydantic `BaseModel` is used for consistency with QEP-030
and future serialization capability, but field processors are intentionally NOT equipped
with JSON discriminator tags — their hierarchy is smaller and they are not pipeline-
reconstructible like ODMR processors. This is a deliberate architectural departure;
if field processor serialization becomes necessary, upgrade the hierarchy to match
QEP-030's pattern with `type` literals and `ProcessorSpec` unions.

---

### 3.2  `HotPixelFilter`

Matches QDMlab's `filter_hot_pixels`. Two detection modes:

**Value-based** (default): a pixel is "hot" if `|value| > median + threshold_sigma * std`.

**Chi2-based**: when `chi2_map` is provided to `process()`, use it for detection instead.
Pixels with poor fit quality are filtered regardless of their field value. This is often
better because a bad-fit pixel may produce a plausible-looking field value by coincidence.

```python
class HotPixelFilter(BaseFieldProcessor):
    threshold_sigma: float = 5.0
    window_size: int = 3              # half-window radius (total window = 2w+1 square)
    replacement: Literal['mean', 'nan', 'zero'] = 'mean'
    absolute_threshold: float | None = None  # always filter |field| > this value first

    def process(
        self,
        field_map: NDArray,
        pixel_spacing: float,
        chi2_map: NDArray | None = None,
    ) -> NDArray:
        """Filter hot pixels.

        Args:
            field_map: (H, W) field values.
            pixel_spacing: Unused (for API consistency).
            chi2_map: Optional (H, W) chi-squared values for detection instead of field.

        Returns:
            (H, W) filtered field map.
        """
```

**Algorithm:**

```
detection_source = chi2_map if chi2_map is not None else field_map

1. Pre-filter absolute outliers (|value| > absolute_threshold → NaN temporarily)
2. Compute median and std of detection_source ignoring NaN
3. Build outlier mask: |detection_source| > median + threshold_sigma * std
4. For each outlier pixel in field_map:
     neighbours = field_map[row-w:row+w+1, col-w:col+w+1]
     if replacement == 'mean':
         set pixel = nanmean(neighbours excluding centre)
     elif replacement == 'nan':  set pixel = NaN
     elif replacement == 'zero': set pixel = 0.0
5. Log: N filtered / M total = P%
```

Edge pixels: clip window to image boundary (as in QDMlab).

---

### 3.3  `QuadraticBackgroundSubtractor`

Fits a polynomial surface to the field map and subtracts it. Matches `QuadBGsub`.

```python
@dataclass
class QuadraticBackgroundSubtractor(BaseFieldProcessor):
    degree: int = 2              # 1 = linear plane, 2 = quadratic surface
    mask: NDArray | None = None  # (H, W) bool — exclude these pixels from the fit
                                 # e.g. the sample region, to fit background only
```

**Algorithm:**

```
Build design matrix A with polynomial features up to `degree`:
    degree=1: [1, x, y]
    degree=2: [1, x, y, x², xy, y²]
    (normalise x, y to [-1, 1] for numerical stability)

Solve: coeffs = lstsq(A[~mask], field[~mask])
Surface = A @ coeffs  (evaluated at all pixels)
Output = field - surface
```

Using numpy's `np.linalg.lstsq` with `rcond=None`. The mask is *excluded from the fit*
but the surface is subtracted everywhere — this is correct when the mask covers the sample
(you don't want the sample's field to bias the background estimate).

---

### 3.4  `UpwardContinuation`

Continues the measured field map to a virtual measurement plane `dz` metres above the
source. Matches `UpCont.m`.

The physics: in free space, a field component B satisfies Laplace's equation. In Fourier
space, continuing upward by height `dz` multiplies each spatial frequency `k` by
`exp(-dz * k)` — a low-pass filter that suppresses short-range (near-source) features.

```python
@dataclass
class UpwardContinuation(BaseFieldProcessor):
    dz: float                  # continuation height in metres (positive = away from source)
    padding_factor: float = 3.0   # extend image by this factor before FFT (avoid wrap-around)
    oversampling: int = 2         # zero-pad in freq. domain (matches QDMlab FOVERSAMPL)
```

**Algorithm:**

```python
def process(self, field_map: NDArray, pixel_spacing: float) -> NDArray:
    ny, nx = field_map.shape
    # 1. Zero-pad: embed field_map at centre of a (padding_factor * ny, ...) array
    py, px = int(ny * self.padding_factor), int(nx * self.padding_factor)
    padded = np.zeros((py, px))
    oy, ox = (py - ny) // 2, (px - nx) // 2
    padded[oy:oy+ny, ox:ox+nx] = field_map

    # 2. Build spatial frequency grid
    fs = 1.0 / pixel_spacing
    ny_fft = py * self.oversampling
    nx_fft = px * self.oversampling
    fy = np.fft.fftfreq(ny_fft, d=pixel_spacing)  # cycles/m
    fx = np.fft.fftfreq(nx_fft, d=pixel_spacing)
    Fx, Fy = np.meshgrid(fx, fy)
    k = 2 * np.pi * np.sqrt(Fx**2 + Fy**2)        # rad/m

    # 3. Apply continuation filter
    H = np.exp(-self.dz * k)
    F = np.fft.fft2(padded, s=(ny_fft, nx_fft))
    out_full = np.real(np.fft.ifft2(F * H))

    # 4. Crop back (only take non-oversampled portion, then un-pad)
    out = out_full[:py, :px]
    return out[oy:oy+ny, ox:ox+nx]
```

Note: `dz > 0` attenuates high spatial frequencies (low-pass, moves field source farther).
`dz < 0` is downward continuation — amplifies noise, use with caution.

---

### 3.5  `BlankSubtractor`

Subtracts a "blank" (diamond without sample) field map from the measurement. Used to
remove the intrinsic contribution of the diamond and experimental setup. Matches
`subtract_blank.m`.

```python
@dataclass
class BlankSubtractor(BaseFieldProcessor):
    blank_map: NDArray                       # (H, W) pre-computed blank field map
    alignment: AlignmentTransform | None = None  # if blank needs spatial alignment
```

If `alignment` is provided, the blank map is warped into the measurement frame before
subtraction. This handles the case where the blank was measured in a different session
and the diamond shifted.

```python
def process(self, field_map: NDArray, pixel_spacing: float) -> NDArray:
    blank = self.blank_map
    if self.alignment is not None:
        blank = self.alignment.apply(blank, field_map.shape)
    return field_map - blank
```

---

### 3.6  `B111ToBzConverter`

Converts a B111 map (field component along the NV [111] axis) to Bz (vertical component)
using a frequency-domain rotation. Matches `B111ToBz_CommLine.m`.

The [111] direction makes an angle θ ≈ 54.7° with the z-axis. In the frequency domain,
the relationship between field components is determined by the tensor that maps Bz → B111.

```python
@dataclass
class B111ToBzConverter(BaseFieldProcessor):
    theta: float = 54.7356    # NV tilt angle from z-axis, degrees
    phi: float = 0.0          # NV azimuth angle, degrees
    padding_factor: float = 3.0
```

The full derivation follows MIT paleomag group formalism; implementation deferred to the
coding phase with reference to `MITBxByFromBz.m`.

---

### 3.7  `FieldProcessingPipeline`

```python
class FieldProcessingPipeline:
    def __init__(self) -> None:
        self.processors: list[BaseFieldProcessor] = []

    def add(self, processor: BaseFieldProcessor) -> FieldProcessingPipeline:
        """Append a processor. Returns self for chaining."""
        self.processors.append(processor)
        return self

    def process(self, field_map: NDArray, pixel_spacing: float) -> NDArray:
        """Apply processors in order. Each receives the output of the previous."""
        result = field_map.copy()
        for proc in self.processors:
            result = proc.process(result, pixel_spacing)
            logger.debug(f'{proc.__class__.__name__} applied')
        return result

    def process_result(
        self,
        fit_result: FitResult,
        component: str = 'remanent',
    ) -> NDArray:
        """Extract B111 component from FitResult and process it.

        Args:
            fit_result: FitResult with b111 xr.Dataset (post-QEP-025).
            component: 'remanent' or 'induced'.

        Returns:
            (H, W) processed field map as NDArray.
        """
        field_map = fit_result.b111[component].values  # xr.DataArray → NDArray (H, W)
        return self.process(field_map, fit_result.pixel_spacing)

    def process_stack(
        self,
        stack: ResultStack,
        component: str = 'remanent',
    ) -> list[NDArray]:
        """Apply pipeline to every entry in a ResultStack."""
        return [
            self.process(stack.get_b111(i, component), stack.entries[i].result.pixel_spacing)
            for i in range(len(stack.entries))
        ]
```

---

## Typical Usage

```python
from QDMpy.field_processing import (
    FieldProcessingPipeline, HotPixelFilter,
    QuadraticBackgroundSubtractor, UpwardContinuation,
)

pipeline = (
    FieldProcessingPipeline()
    .add(HotPixelFilter(threshold_sigma=5.0))
    .add(QuadraticBackgroundSubtractor(degree=2))
    .add(UpwardContinuation(dz=5e-6))
)

bz_processed = pipeline.process_result(fit_result, component='remanent')

# Or process a whole stack
processed_maps = pipeline.process_stack(result_stack)
```

---

## New Module: `QDMpy/field_processing.py`

All classes live in a single new top-level module. No existing modules are modified.

---

## Dependencies

**Required additions to `pyproject.toml`:**

```toml
"scikit-image>=0.21",
```

`scipy.ndimage` and numpy FFT are already available (scipy is a core dependency).

After adding scikit-image, run `uv lock` to update the lock file.

---

## Alternatives Considered

### A. Operate directly on FitResult rather than raw NDArray
Rejected. Processors should be composable on any 2D array, not tightly coupled to
`FitResult`. A user may want to process a Bz map loaded from a .mat file directly.

### B. Inplace mutation of input arrays
Rejected. Processors always copy — easier to debug and avoids aliasing bugs in pipelines.

### C. Combine UpwardContinuation and B111ToBz into one step
Rejected. They are physically and algorithmically distinct. Upward continuation is a
height correction; coordinate rotation is a geometric re-projection. Both may be needed
independently.
