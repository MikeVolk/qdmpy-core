# QEP-028 — Demagnetization Behavior Analysis

**Status:** Draft
**Created:** 2026-02-18
**Depends on:** QEP-026 (ResultStack), QEP-027 (FieldProcessingPipeline)

---

## Motivation

The central experiment in paleomagnetic QDM is the *demagnetization series*: the same
sample is measured after progressive demagnetization (thermal or AF steps) or after
applying increasing fields. By tracking how the magnetic signal evolves across steps,
one can extract coercivity spectra, identify magnetically distinct domains, and estimate
total magnetic moment as a function of applied field.

QDMlab's `demag_behavior.m` implements this in ~300 lines. It:

1. Aligns all maps to a reference (`get_transformed_maps`)
2. Has the user draw ROI masks on the reference map
3. For each map × ROI: counts pixels above threshold, computes their sum, compares to reference
4. Bootstraps error estimates by randomly shifting the mask ±N pixels

The output is a `(n_roi, n_files, 2)` array of `(value, std)` pairs for each metric.

QDMpy has none of this. This QEP introduces a clean, data-oriented API with xarray output.

---

## Goals

1. `DemagROI` — a named region-of-interest mask on a field map.
2. `DemagMetrics` — the per-(roi, entry) metrics from one measurement.
3. `DemagAnalysis` — orchestrates alignment, processing, masking, and metric computation.
4. Structured `xr.Dataset` output with labelled coordinates for downstream analysis.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### 3.1  `DemagROI`

```python
@dataclass
class DemagROI:
    mask: NDArray          # (H, W) bool — True inside the selected region
    label: str             # human-readable name, e.g. 'grain_A', 'background'
    threshold: float = 0.25  # fraction of max field value used to define
                             # the "active" sub-region within the ROI
```

The ROI and the "active mask" are distinct concepts (matching QDMlab):

- `mask` — the full hand-drawn region (rectangular or freehand).
- **Active pixels** — pixels within `mask` where `|field| > threshold * max(|field|)`.
  Only active pixels are counted; the full mask is used for the bootstrap shift domain.

ROIs can be created interactively (future CLI/GUI integration) or programmatically from
a boolean array:

```python
roi = DemagROI.from_threshold(reference_map, threshold=0.3, label='grain_A')
roi = DemagROI(mask=my_mask_array, label='grain_A')
```

---

### 3.2  `DemagMetrics` (per roi × entry)

```python
@dataclass
class DemagMetrics:
    n_pixels: int          # total active pixels in reference
    pos_pixels: float      # active pixels with positive sign (mean over bootstrap)
    neg_pixels: float      # active pixels with negative sign (mean over bootstrap)
    pos_pixel_ratio: float # pos_pixels / n_pixels
    mask_sum: float        # sum of field values in active region (proportional to moment)
    sum_diff: float        # sum(current - reference) in active region
    mse: float             # mean squared error vs reference

    pos_pixels_std: float      = 0.0   # bootstrap std
    neg_pixels_std: float      = 0.0
    pos_pixel_ratio_std: float = 0.0
    mask_sum_std: float        = 0.0
    sum_diff_std: float        = 0.0
    mse_std: float             = 0.0
```

---

### 3.3  `DemagAnalysis`

```python
class DemagAnalysis:
    def __init__(
        self,
        stack: ResultStack,
        pipeline: FieldProcessingPipeline | None = None,
        component: str = 'remanent',
        bootstrap_n: int = 1,
        pixel_shift: int = 4,
    ) -> None:
        """
        Args:
            stack: Aligned ResultStack (compute_alignments must have been called).
            pipeline: Optional field processing to apply before analysis.
                      Typical: HotPixelFilter → QuadBGsub → UpwardContinuation.
            component: Which B111 component to analyse ('remanent' or 'induced').
            bootstrap_n: Number of bootstrap iterations for error estimation.
                         1 = no error estimate (fast). >1 = random mask shifts.
            pixel_shift: Max pixel displacement for bootstrap mask shifts.
        """

    def add_roi(self, roi: DemagROI) -> None: ...

    def run(self) -> xr.Dataset:
        """Execute the full analysis and return a structured Dataset.

        Returns:
            xr.Dataset with dimensions (roi, entry) and variables:
                - pos_pixels, neg_pixels, pos_pixel_ratio
                - mask_sum, sum_diff, mse
                - *_std variants for each when bootstrap_n > 1
            Coordinates:
                - roi: [roi.label for roi in self.rois]
                - entry: [e.label for e in stack.entries]
                - entry_metadata: structured array of per-entry metadata
        """
```

---

### 3.4  Analysis Algorithm

```
For each entry i in stack:
    1. Retrieve aligned field map: maps[i] = stack.get_b111(i, component)
    2. Apply pipeline: maps[i] = pipeline.process(maps[i], pixel_spacing)

reference_map = maps[reference_idx]

For each ROI r:
    For each entry i:
        metrics[r, i] = _compute_metrics(reference_map, maps[i], roi=r, n=bootstrap_n)

Return as xr.Dataset.
```

**`_compute_metrics(reference, target, roi, n)`:**

```
active_ref = reference[roi.mask & (|reference| > roi.threshold * max(|reference|))]

for b in range(n):
    if n > 1:
        shifted_mask = shift_mask(roi.mask, dx=randint(-pixel_shift, pixel_shift),
                                             dy=randint(-pixel_shift, pixel_shift))
    else:
        shifted_mask = roi.mask

    active = target[shifted_mask & active_condition]
    n_pixels[b]   = len(active_ref)
    pos[b]        = (active > 0).sum()
    neg[b]        = (active < 0).sum()
    mask_sum[b]   = active.sum()
    sum_diff[b]   = (active - active_ref).sum()
    mse[b]        = mean_squared_error(active, active_ref)

return DemagMetrics(
    n_pixels = n_pixels.mean(),
    pos_pixels = pos.mean(), pos_pixels_std = pos.std(),
    ...
)
```

The `shift_mask` helper translates the boolean mask by `(dy, dx)` pixels using
`scipy.ndimage.shift` with `order=0` (nearest-neighbour, so the mask stays binary).

---

### 3.5  Output Structure

```python
ds = analysis.run()

# Access by label
grain_a_ratio = ds['pos_pixel_ratio'].sel(roi='grain_A')  # (n_entries,) DataArray
ten_mt_sum    = ds['mask_sum'].sel(entry='10mT')           # (n_rois,) DataArray

# Plot coercivity curve
ds['mask_sum'].sel(roi='grain_A').plot()

# Export to pandas for further analysis
df = ds.to_dataframe()
```

---

## Typical Usage

```python
from QDMpy.result_stack import ResultStack
from QDMpy.field_processing import FieldProcessingPipeline, HotPixelFilter, UpwardContinuation
from QDMpy.demag import DemagAnalysis, DemagROI
import numpy as np

# Build stack from FitResults
stack = ResultStack()
for label, fit_result, led, laser in measurements:
    stack.add(fit_result, led, laser, label=label)

# Align all entries to the first
stack.compute_alignments(method='phase_correlation')
stack.save_transforms('/data/FOV1/transforms.json')

# Post-processing pipeline
pipeline = (
    FieldProcessingPipeline()
    .add(HotPixelFilter(threshold_sigma=5.0))
    .add(UpwardContinuation(dz=5e-6))
)

# Define ROIs
reference_map = stack.get_b111(0, component='remanent')
roi_a = DemagROI.from_threshold(reference_map, threshold=0.3, label='grain_A')
roi_b = DemagROI(mask=manual_mask, label='background')

# Run analysis
analysis = DemagAnalysis(stack, pipeline=pipeline, bootstrap_n=50, pixel_shift=4)
analysis.add_roi(roi_a)
analysis.add_roi(roi_b)

results = analysis.run()
print(results)
# <xr.Dataset>
# Dimensions: (roi: 2, entry: 6)
# Coordinates:
#   * roi:   ['grain_A', 'background']
#   * entry: ['NRM', '5mT', '10mT', '20mT', '30mT', '50mT']
# Variables:
#     pos_pixel_ratio     (roi, entry) float64
#     pos_pixel_ratio_std (roi, entry) float64
#     mask_sum            (roi, entry) float64
#     ...
```

---

## New Module Structure

```
QDMpy/
├── result_stack.py        # ResultStack, StackEntry, AlignmentTransform (QEP-026)
├── field_processing.py    # Pipeline + processors (QEP-027)
└── demag.py               # DemagAnalysis, DemagROI, DemagMetrics (QEP-028)
```

---

## Alternatives Considered

### A. Return a plain dict of numpy arrays instead of xr.Dataset
Rejected. Named coordinates (roi label, entry label) are essential for interpretability.
The whole point of the refactor is to make data self-describing.

### B. Compute alignment inside DemagAnalysis rather than requiring a pre-aligned stack
Rejected. Alignment is an expensive, potentially interactive step that should persist
across analysis runs. Requiring a `ResultStack` with pre-computed alignments makes the
dependency explicit and cacheable.

### C. Use a pandas DataFrame instead of xr.Dataset
Rejected. The (roi, entry) structure is naturally 2D; xarray handles it better than a
flat DataFrame and integrates with existing xarray-based outputs elsewhere in QDMpy.

### D. Separate positive and negative ROIs (as QDMlab does)
QDMlab uses two separate masks — one for positive pixels, one for negative — to analyse
bidirectional demagnetization. This is captured here through the signed nature of
`pos_pixels` and `neg_pixels` in `DemagMetrics`, with the threshold applied to
`|field| > threshold * max(|field|)` rather than `field > 0`. This correctly counts
both polarities without requiring the user to draw two masks.
