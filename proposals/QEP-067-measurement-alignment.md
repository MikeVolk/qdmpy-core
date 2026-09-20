# QEP-067 -- Measurement Alignment and Stitching

| Field   | Value |
|---------|-------|
| Status  | Draft |
| Created | 2026-03-22 |
| Scope   | New module `qdmpy.alignment`, new container `Mosaic`, extensions to `qdmpy.io` |
| Depends | — |
| Related | QEP-025 (semantic coordinates) |
| Revised | 2026-03-22 | Field naming alignment (contrasts/linewidths), Mosaic uses frozen BaseModel |

## Motivation

QDM measurements often cover only part of a sample. When the sample is
repositioned under the microscope and re-measured, the two datasets need to be
spatially registered and composited into a single unified view. Currently there
is no alignment or stitching capability in qdmpy-core — users must do this
manually in external tools, losing integration with the fitting pipeline and
.qdm serialization.

## Problem Statement

Given two `QDMResult` objects whose spatial extents overlap (the same sample
region, measured after repositioning), there is no way to:

1. Compute the spatial transform that maps one measurement onto the other.
2. Apply that transform across all derived 2D data spaces consistently.
3. Composite the overlapping region into a single, larger canvas.
4. Serialize and reload the stitched result.

## Goals

1. Compute an affine transform between two measurements using multi-channel
   registration (light image + B111 remanent + B111 induced).
2. Warp and composite all derived 2D maps (light, B111 remanent, B111 induced,
   chi2, contrasts, linewidths, and optional Bxyz) onto a common canvas.
3. Provide a `Mosaic` container that holds the stitched maps and tracks
   provenance (which measurements contributed, what transform was applied).
4. Serialize `Mosaic` to the .qdm format (or a `.qdm-mosaic` variant).
5. Exclude laser images from alignment and stitching (per user requirement).
6. Start with pairwise stitching (2 measurements); design the interface so
   extending to N measurements is straightforward.

## Non-goals

- **Raw ODMR spectra stitching.** The 5D spectral arrays are too large and
  their frequency axes may differ. Only derived 2D result maps are in scope.
- **Lens distortion correction.** Barrel/pincushion distortion at microscope
  edges is a real concern but is deferred to a follow-up QEP once the core
  alignment pipeline exists.
- **True perspective (homography) correction.** Small trapezoidal distortions
  from imperfect sample mounting are approximated by affine for now. The
  transform abstraction is designed so homography can replace affine later.
- **Automatic measurement discovery.** The user explicitly provides the two
  measurements to align; no folder-scanning or batch mode.
- **GUI integration.** This QEP covers the core library API only. A GUI QEP
  for interactive alignment (ROI selection, transform preview) will follow.

## Proposed Design

### Architecture

```
qdmpy/
  alignment/
    __init__.py          # public API: align, stitch
    _registration.py     # multi-channel affine registration
    _warping.py          # affine warp + canvas compositing
    _mosaic.py           # Mosaic container
```

### Registration pipeline (`_registration.py`)

**Input:** Two `QDMResult` objects (reference and moving).

**Channel extraction:**
Extract three 2D arrays from each result:
- `light_image`
- `b111_remanent`
- `b111_induced`

Each channel is independently normalized to [0, 1] before registration.

**Per-channel transform estimation:**
For each channel pair, estimate an affine transform using phase
cross-correlation (translation) followed by least-squares affine refinement.
The method:

1. **Coarse alignment:** `scipy.ndimage` or `skimage.registration.phase_cross_correlation`
   for sub-pixel translation estimate.
2. **Fine refinement:** Extract matching feature points (ORB or similar via
   scikit-image) and fit an affine matrix with RANSAC to reject outliers.

If scikit-image is too heavy a dependency, a fallback path using only
`scipy.ndimage` (phase correlation for translation + optional rotation search)
is provided, at the cost of no shear/scale correction.

**Multi-channel fusion:**
Each channel produces a candidate affine matrix. The final transform is
computed as the **weighted average of the affine parameters**, where weights
are the registration confidence scores (e.g., peak correlation value). This
avoids the failure mode where one channel (e.g., a featureless light image)
dominates.

**Output:** `AffineTransform` dataclass holding the 2x3 affine matrix, per-channel
confidences, and the fused confidence score.

```python
@dataclass(frozen=True)
class AffineTransform:
    matrix: NDArray           # shape (2, 3) affine matrix
    channel_scores: dict[str, float]  # per-channel confidence
    confidence: float         # fused confidence (weighted mean)
```

### Warping and compositing (`_warping.py`)

**Canvas computation:**
Given the reference result dimensions `(H1, W1)` and the affine-transformed
bounding box of the moving result, compute the output canvas size `(H_out, W_out)`
and the offset that places the reference at its correct position.

**Warping:**
Apply `scipy.ndimage.affine_transform` to each 2D map of the moving result,
resampling onto the output canvas grid. Use order=1 (bilinear) for speed;
order=3 (bicubic) as an option.

**Compositing:**
In the overlap region, combine values using a configurable strategy:

| Strategy | Description |
|----------|-------------|
| `"average"` | Simple mean of overlapping pixels (default) |
| `"reference"` | Keep reference values, fill gaps with moving |
| `"moving"` | Keep moving values, fill gaps with reference |

Future strategies (distance-weighted blending, chi2-weighted) are out of scope
but the `CompositeStrategy` type alias makes extension trivial.

**Mask tracking:**
Each source measurement contributes a boolean coverage mask. The composite
tracks a `coverage: NDArray` of shape `(H_out, W_out)` with values 0 (no data),
1 (single source), or 2 (overlap).

### Mosaic container (`_mosaic.py`)

```python
class Mosaic(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    light_image: NDArray              # (H_out, W_out)
    b111_remanent: NDArray            # (H_out, W_out), uT
    b111_induced: NDArray             # (H_out, W_out), uT
    chi2: NDArray                     # (H_out, W_out)
    contrasts: NDArray                # (H_out, W_out)
    linewidths: NDArray               # (H_out, W_out)
    bxyz: dict[str, NDArray] | None   # optional {bx, by, bz, btotal}
    coverage: NDArray                 # (H_out, W_out), uint8: 0/1/2
    pixel_spacing: float              # metres (must match between inputs)
    scan_dimensions: tuple[int, int]  # (H_out, W_out)
    transform: AffineTransform        # the computed alignment
    composite_strategy: str           # strategy used
    source_metadata: tuple[dict, dict]  # metadata from each input
```

`Mosaic` is **immutable** (frozen Pydantic `BaseModel`, same pattern as
`QDMResult`). It is intentionally NOT a `QDMResult` subclass — it has
different semantics (no ODMR data, no single model_name, no single FitResult).
Field names (`contrasts`, `linewidths`) match `QDMResult`'s property names for
consistency. Shared plotting utilities can accept `QDMResult | Mosaic` via a
protocol or duck typing on the common fields.

### Public API

```python
from qdmpy.alignment import align, stitch

# Step 1: compute the transform
transform = align(reference=result_a, moving=result_b)
# -> AffineTransform

# Step 2: apply it
mosaic = stitch(
    reference=result_a,
    moving=result_b,
    transform=transform,
    strategy='average',
)
# -> Mosaic
```

Separating `align` and `stitch` lets users inspect/override the transform
before applying it.

### Validation

- Both inputs must have the same `pixel_spacing` (raise `DataValidationError`
  if not).
- Both inputs must have `light_image` and `b111_remanent`/`b111_induced`
  available (raise if missing — e.g., unfitted result).
- The fused confidence score is returned so users can warn on low-confidence
  alignment.

## Decisions

### D1: Multi-channel registration over single-channel

**Context:** Light images can be featureless for some samples. B111 maps have
strong spatial features from magnetic sources but are noisier.

**Decision:** Use light + B111 remanent + B111 induced simultaneously, fusing
per-channel affine estimates by confidence-weighted averaging.

**Consequences:** More robust alignment across diverse samples. Slightly more
compute (3x registration), but these are 2D maps (~2k x 2k), so cost is
negligible.

**Rejected:** Single-channel (light only) — fails on featureless samples.
Single-channel (B111 only) — noisier, less reliable for translation.

### D2: Affine transform model

**Context:** Sample repositioning introduces translation and rotation. Small
trapezoidal distortions can occur from imperfect mounting. Full lens distortion
correction is deferred.

**Decision:** Affine (6 DOF: translation, rotation, scale, shear). Covers the
common cases without the complexity of homography (8 DOF).

**Consequences:** Small perspective distortions are approximated, not corrected
exactly. Acceptable for typical QDM workflows.

**Rejected:** Rigid (too restrictive — no shear/scale). Homography (overkill
for this stage; adds complexity in compositing).

### D3: New container over extending QDMResult

**Context:** The stitched result lacks ODMR spectra, has no single model_name,
and has different spatial dimensions than either input.

**Decision:** New `Mosaic` as a frozen Pydantic `BaseModel`, separate from
`QDMResult`. Uses the same `ConfigDict(frozen=True, arbitrary_types_allowed=True)`
pattern as `QDMResult` for consistency (validation, `.model_dump()`, NDArray
support). Field names (`contrasts`, `linewidths`) match `QDMResult`'s properties.

**Consequences:** No risk of breaking existing `QDMResult` consumers. Plotting
code needs minor updates to accept both types. Serialization needs a new
format variant. Consistent Pydantic API across result types.

**Rejected:** Subclassing `QDMResult` — Liskov violation (Mosaic can't provide
`fit_result`, `odmr`, etc.). Plain frozen dataclass — unnecessary divergence
from the established Pydantic `BaseModel` pattern.

### D4: Dependency choice

**Context:** scikit-image provides robust feature detection (ORB), RANSAC, and
affine estimation. scipy alone can handle phase correlation but not
feature-based refinement.

**Decision:** Add `scikit-image` as an **optional dependency**
(`qdmpy-core[alignment]`). The registration module imports it at call time
and raises `ImportError` with install instructions if missing.

**Consequences:** No impact on users who don't use alignment. Adds ~30 MB to
install for those who do.

**Rejected:** Hard dependency (bloats base install). scipy-only (no robust
affine refinement without feature matching).

## Alternatives Considered

### A1: Manual point-based alignment

User clicks corresponding points in both images, transform is fitted from
point pairs. Simpler to implement but requires GUI interaction — not suitable
for a core library API. Could be added as a GUI feature later that feeds
point pairs into the same affine fitting code.

### A2: Stitching raw ODMR spectra

Would enable re-fitting the stitched dataset as a whole. Rejected because:
- 5D arrays are very large (~2k x 2k x 50 freqs x 2 pol x 2 frange = ~800M floats)
- Frequency axes may differ between measurements
- The user confirmed maps-only scope

### A3: Overlap compositing via chi2 weighting

Use per-pixel chi2 to weight which measurement contributes more in the overlap
region. Scientifically motivated but adds complexity. Deferred — the strategy
parameter makes this easy to add later.

## Implementation Steps

### Phase 1: Registration core

**Goal:** Compute affine transform from two QDMResult objects.

**Files:**
- `src/qdmpy/alignment/__init__.py` — public `align()` function
- `src/qdmpy/alignment/_registration.py` — channel extraction, per-channel
  phase correlation + affine refinement, multi-channel fusion
- `tests/alignment/test_registration.py` — synthetic shifted/rotated arrays,
  verify sub-pixel accuracy

**Validation:** `align()` returns correct transform for known synthetic shifts
with < 0.5 pixel RMS error.

### Phase 2: Warping and compositing

**Goal:** Apply transform and composite maps onto a unified canvas.

**Files:**
- `src/qdmpy/alignment/_warping.py` — canvas computation, affine warp,
  composite strategies, coverage mask
- `tests/alignment/test_warping.py` — verify canvas size, overlap averaging,
  coverage mask values

**Depends on:** Phase 1 (needs AffineTransform).

**Validation:** Stitched output has correct dimensions; overlap region values
equal the mean of inputs (for average strategy); coverage mask is accurate.

### Phase 3: Mosaic container

**Goal:** Frozen dataclass holding all stitched maps + provenance.

**Files:**
- `src/qdmpy/alignment/_mosaic.py` — `Mosaic` dataclass
- `src/qdmpy/alignment/__init__.py` — public `stitch()` function
- `tests/alignment/test_mosaic.py` — construction, immutability, field access

**Depends on:** Phase 2 (needs warp/composite output).

### Phase 4: Serialization

**Goal:** Save/load `Mosaic` to `.qdm` HDF5 format.

**Files:**
- `src/qdmpy/io/qdm.py` — extend with `save_mosaic()` / `load_mosaic()`
  (or new `.qdm-mosaic` group structure)
- `tests/io/test_qdm_mosaic.py` — round-trip save/load

**Depends on:** Phase 3.

**Validation:** Round-trip `Mosaic` -> HDF5 -> `Mosaic` preserves all arrays
and metadata within float32 tolerance.

### Phase 5: Integration and documentation

**Goal:** End-to-end example, docstrings, API reference.

**Files:**
- `examples/align_two_measurements.py` — runnable demo
- `docs/tutorials/alignment.md` — narrative tutorial
- Docstrings on all public functions

**Depends on:** Phases 1-4.

## GUI Integration Requirements

1. **Core API contracts the GUI depends on:**
   - `align(reference, moving) -> AffineTransform`
   - `stitch(reference, moving, transform, strategy) -> Mosaic`
   - `Mosaic` fields: `light_image`, `b111_remanent`, `b111_induced`, `chi2`,
     `contrasts`, `linewidths`, `coverage`, `pixel_spacing`, `scan_dimensions`
   - `save_mosaic()` / `load_mosaic()` in `qdmpy.io`

2. **GUI-side work (separate QEP):**
   - Load two measurements, trigger alignment, preview transform overlay
   - Display `Mosaic` maps in the image panel (same rendering as QDMResult)
   - Composite strategy selector (dropdown)
   - Coverage map visualization (highlight overlap region)

3. **No migration required:** This is a new capability; existing sessions and
   configs are unaffected.

4. **GUI acceptance checks:**
   - Load two overlapping measurements -> align -> stitch -> verify mosaic
     renders in image panel with correct spatial extent
   - Verify coverage overlay shows overlap region

## Acceptance Criteria

- [ ] `align()` returns sub-pixel accurate affine transform for synthetic test
      data with known shift + rotation (< 0.5 px RMS)
- [ ] `stitch()` produces a `Mosaic` with correct canvas dimensions and
      composited maps for all data spaces (light, B111 remanent, B111 induced,
      chi2, contrasts, linewidths)
- [ ] Laser image is excluded from alignment and stitching
- [ ] Overlap region uses simple average by default
- [ ] `Mosaic` round-trips through HDF5 serialization
- [ ] Coverage mask correctly identifies no-data (0), single-source (1), and
      overlap (2) regions
- [ ] Both inputs validated for matching `pixel_spacing`
- [ ] scikit-image is optional — `ImportError` with clear message if missing
- [ ] 80%+ test coverage on `qdmpy.alignment`

## Open Questions

1. **Pixel spacing mismatch:** If two measurements have different pixel
   spacings (e.g., different objectives), should we resample one to match?
   Or reject outright? Current proposal rejects — resampling adds complexity
   and is a rare case.

2. **N-way stitching API:** The current design is pairwise. For N measurements,
   should `stitch()` accept a list and chain pairwise, or should there be a
   separate `stitch_multi()` that computes a global alignment graph? Deferred
   to usage feedback.

3. **Confidence threshold:** Should `stitch()` refuse to proceed if the
   alignment confidence is below some threshold, or always produce output
   with a warning? Current proposal: always produce output, log a warning
   if confidence < 0.5.
