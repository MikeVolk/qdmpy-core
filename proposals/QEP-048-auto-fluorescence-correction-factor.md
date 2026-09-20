# QEP-048 — Auto-Detection of Fluorescence Correction Factor

**Status:** Revised Draft (post-investigation)
**Created:** 2026-02-21
**Revised:** 2026-02-21

---

## Motivation

`FluorescenceCorrectionProcessor` requires the user to supply a
`correction_factor` (default 0.2). This scalar is not physically derived and
can be wrong by a factor of several for real samples.

**Original hypothesis:** jointly fit an ESR model and `α` on the "strongest
pixel" to auto-detect the correction factor.

**Investigation result** (see `notebooks/experiments/fluorescence-correction-auto-alpha.ipynb`):
the original approach does not work, and the investigation revealed deeper
problems with the current QDMpy implementation. This proposal is revised based
on those findings and on reading the reference MATLAB implementation in
`~/git/QDMlab`.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Investigation Setup

All experiments were run in
`notebooks/experiments/fluorescence-correction-auto-alpha.ipynb`
against real data from `tests/data/MIL2_FOV1` (2-polarity, 2-freq-range,
1200×1920 pixels, spatially binned 4×4 to 300×480 for speed).

### What "fluorescence correction" means here

`FluorescenceCorrectionProcessor` subtracts a scaled version of the
**baseline-corrected spatial mean spectrum** from every pixel:

```python
baseline_corrected_mean = mean_data - edge_baseline   # shape (n_freq,)
corrected = raw - alpha * baseline_corrected_mean
```

`baseline_corrected_mean` is ≈ 0 at off-resonance frequencies and negative
at ESR dip positions. So `alpha * baseline_corrected_mean` is a component
shaped like the mean ODMR dip.

### What we measured as the "fluorescence signature"

For each spatially binned pixel we computed:
- **off-resonance baseline**: mean of the first and last 4 frequency points
- **ESR dip depth**: `off_res_baseline − min(spectrum)`

We used `corr(baseline, dip_depth)` across all pixels as a scalar diagnostic.
Expected physics: if some pixels have more non-NV fluorescence, they have a
higher absolute off-resonance signal **and** shallower ODMR dips (fluorescence
dilutes the contrast) — producing a **negative** correlation.

### Methods tested for auto-detecting α

**Method A — single-pixel joint fit:** fit
`pixel(f) = ESR14N(f; θ) + α · mean_corrected(f)` simultaneously for 7
parameters (6 ESR + α) using `scipy.optimize.curve_fit`.

**Method B — single-pixel profile likelihood:** for each α ∈ [0, 1] in 40–80
steps, subtract `α · mean_corrected` from the strongest pixel, fit ESR14N to
the residual, record chi². Find the α minimising chi².

**Method C — multi-pixel sum-chi²:** same profile likelihood sweep but sum
chi² across the top-10 strongest pixels (ranked by L2 deviation from mean).

**Method D — correlation minimisation:** for each α, apply the correction to
all pixels and compute `corr(baseline, dip_depth)`. Find α minimising `|corr|`.

**Synthetic validation:** injected a known fluorescence contribution
`α_true · mean_corrected` into the strong pixel and tested recovery
(α_true ∈ {0.0, 0.1, 0.2, 0.35, 0.5, 0.7}).

---

## Empirical Findings (MIL2_FOV1 dataset)

### 1. Fluorescence is present and spatially non-uniform

Raw MATLAB data (before any normalization by QDMpy):

```
corr(off-resonance baseline, ESR dip depth) = −0.43
```

Negative correlation = fluorescence signature: pixels with higher
off-resonance signal have shallower ODMR dips, consistent with non-NV
fluorescence diluting the contrast.

### 2. QDMpy's max-normalization hides the fluorescence signal

After `NormalizationProcessor` (divide each pixel by its own **max**):

```
corr(baseline, dip depth) drops from −0.43 → −0.24
```

Max-normalization forces every pixel's off-resonance level to exactly 1.0,
absorbing pixel-specific absolute fluorescence. The residual −0.24 shows
spatial variation remains, but the primary baseline information is lost.

### 3. The current correction pipeline is applied in the wrong order

Applying `FluorescenceCorrectionProcessor` **after** `NormalizationProcessor`
makes the baseline–dip correlation **worse** (more negative) for every α > 0.
The current `from_folder()` order — normalize then correct — moves in the
wrong direction.

### 4. The mean-subtraction formula targets the wrong artifact

The current formula removes a component shaped like the **mean ODMR dip**.
This is appropriate for ODMR-shaped cross-talk, not for a flat non-resonant
fluorescence background. For a flat, spatially uniform fluorescence the correct
inverse is:

```
I_corrected(f) = (I_norm(f) − α) / (1 − α)
```

### 5. Globally uniform fluorescence is undetectable from max-normalized data

If F is the same for every pixel, max-normalization absorbs it completely.
Any α can be absorbed by the ESR contrast and offset parameters. Auto-detecting
a globally uniform fluorescence requires an external reference measurement.

### 6. Multi-pixel approach improves robustness, not the degeneracy

The top-N sum-chi² approach is more robust against outliers than single-pixel
fitting. However, on max-normalized data it still returns α=0 due to the same
degeneracy. It is the right strategy once the pipeline and normalization are fixed.

---

## Reference MATLAB Implementation (QDMlab)

Reading `~/git/QDMlab/fitting/correct_global.m` and
`~/git/QDMlab/utilities/prepare_raw_data.m` reveals important differences.

### MATLAB normalises by the **mean across frequencies**, not the max

```matlab
% prepare_raw_data.m, lines 94-99
NormalizationFactor = mean(binData, 3);    % mean over frequency axis
binDataNorm(:,:,y) = binData(:,:,y) ./ NormalizationFactor;
```

This is the critical difference. Mean-normalisation preserves the
off-resonance baseline information:

- **QDMpy max-norm**: forces off-res = 1.0 for every pixel → baseline
  information destroyed → fluorescence correction cannot work.
- **MATLAB mean-norm**: pixels with fluorescence have off-res slightly > 1;
  pixels without have off-res slightly < 1 → baseline variation survives →
  correction can address it.

### MATLAB correction formula is richer (two simultaneous operations)

```matlab
% correct_global.m, line 54
corrected(i,j,k) = (baselinerange / pixelrange) * (data(i,j,k) - meanbaseline) ...
                   - globalFraction * specZBL(k,1) + globalmeanBL;
```

where:
- `globalmeanBL` = edge baseline of the global (spatial mean) spectrum
- `baselinerange = globalmeanBL - min(global spectrum)` = dip depth of global spectrum
- `specZBL = global_spectrum - globalmeanBL` = zero-baseline global spectrum (negative at dips)
- `meanbaseline` = edge baseline of this pixel
- `pixelrange = meanbaseline - min(pixel spectrum)` = dip depth of this pixel

**Operation 1 — amplitude equalisation:**
`(baselinerange / pixelrange) * (pixel - meanbaseline)` rescales each
pixel so that its dip depth matches the global spectrum's dip depth, then
centres it at zero. This normalises away pixel-to-pixel NV density differences.

**Operation 2 — global spectrum subtraction:**
`- globalFraction * specZBL + globalmeanBL` subtracts `globalFraction`
times the zero-baseline global spectrum and re-adds the global baseline.

Together: pixels are first amplitude-equalised (removing NV density variation)
then the common global spectral shape is partially subtracted. QDMpy only does
Operation 2, missing the amplitude equalisation.

### MATLAB pipeline order

```
load → bin → normalize-by-mean → correct_global → fit
```

Correction happens **after** mean-normalization. This works because mean-norm
preserves baseline variation, unlike QDMpy's max-norm.

### globalFraction default and estimation

- Default: **0.5** (vs QDMpy's 0.2)
- `globalFraction_estimator.m` provides an interactive slider UI that
  shows the correction effect on three pixels (leftmost dip, random,
  rightmost dip) — entirely manual, no auto-detection.

### Reference spectrum: per (pol, fr) slice, not a combined mean

`correct_global.m` receives a 3D array `(H, W, n_freq)` for a **single
(pol, fr) slice** and computes `mean(data, [1 2])` — spatial mean of that
slice only. Both QDMpy and MATLAB operate identically here: per slice.

This is physically correct for two reasons:

- `frange_0` and `frange_1` have different frequency axes; combining them into
  a single reference would produce a template with the wrong spectral positions
  for correcting either range alone.
- The two polarities have ESR dips at different absolute frequencies (the
  applied bias field shifts them). Mixing polarities would smear the dip
  positions in the template.

### Physical origin of GF: a zero-field ODMR signal, not flat background

The GF from laser light bouncing inside the diamond is **not a flat background**.
It arises from NV centers in the diamond bulk excited by back-scattered laser
light. These bulk NVs experience approximately zero applied magnetic field and
therefore produce a triple Lorentzian ODMR response clustered around the
zero-field splitting (ZFS, ~2.870 GHz for 14N:
peaks at ~2.8678, 2.8700, 2.8722 GHz).

The sweep ranges are deliberately set by the experimenter to bracket ZFS from
both sides — frange_0 covers the low branch (below ZFS) and frange_1 the high
branch (above ZFS). ZFS therefore always sits at the frange boundary **by
construction**, not by coincidence. The GF zero-field ODMR peaks (fixed at ZFS
regardless of applied field) consequently always appear:

- at the **right edge** of frange_0 (approaching ZFS from below), and
- at the **left edge** of frange_1 (departing ZFS upward).

The sensing NV dips, by contrast, move with the applied field: stronger sample
magnetisation + bias field pushes the ESR further into the interior of each
frange. This has two consequences for GF:

1. **Separation improves with field strength.** Over strongly magnetised regions
   the sensing dips are far from ZFS and spectrally distinct from the GF peak.
   Over weakly magnetised (or zero-field) regions the two overlap, making GF
   contamination hardest to correct.

2. **Edge-baseline is always corrupted.** Because the GF Lorentzian is always
   near the frange edges, the standard edge-baseline estimate (mean of the first
   or last N frequency points) is systematically elevated by the GF signal.
   This biases the apparent dip depth, which is why the correction must be
   applied before fitting — not after the dip depth has already been estimated
   from a corrupted baseline.

---

## Revised Design

### Root cause summary

| Issue | Current QDMpy | MATLAB QDMlab | Fix needed |
|-------|--------------|---------------|-----------|
| Normalization method | max per pixel | mean per pixel | **Remove** max-norm; mean-norm is the only physically valid option |
| Correction formula | subtract `α · baseline_corrected_mean` | amplitude-equalise + subtract `GF · specZBL` | Implement the amplitude equalisation step |
| Default factor | 0.2 | 0.5 | Align with MATLAB after fixing formula |
| Auto-detection | None | Manual slider | Two-component fit on 10×10 patch around strongest pixel |

### Proposed corrections

**Step 1 (required): Remove max-norm from `NormalizationProcessor`** —
max-normalisation absorbs the per-pixel baseline variation that is the only
observable signature of GF in normalised data. Keeping it as an option would
silently invalidate any downstream fluorescence correction or auto-detection.
Replace `method='max'` with `method='mean'` as the sole valid method; raise
`ValueError` if `method='max'` is requested, with a message explaining why.

**Step 2 (required): Fix `FluorescenceCorrectionProcessor`** — implement the
MATLAB amplitude-equalisation term:

```python
def process(self, data: ODMRData) -> ODMRData:
    global_spec = data.data.mean(dim=('y', 'x'))   # (pol, fr, freq)
    global_baseline = _edge_mean(global_spec)       # (pol, fr)
    global_range = global_baseline - global_spec.min('freq_idx')  # dip depth

    pixel_baseline = _edge_mean(data.data)          # (pol, fr, y, x)
    pixel_range = pixel_baseline - data.data.min('freq_idx')

    spec_zbl = global_spec - global_baseline        # zero-baseline global shape

    # Operation 1: amplitude equalise
    amplitude_corrected = (global_range / pixel_range) * (data.data - pixel_baseline)
    # Operation 2: subtract global shape and restore baseline
    corrected = amplitude_corrected - self.correction_factor * spec_zbl + global_baseline

    return ODMRData(data=corrected, metadata=data.metadata.copy())
```

**Step 3 (new feature): Auto-detect `correction_factor`** — use a two-component
fit on the spatial mean spectrum. The GF peaks are pinned to ZFS at known
positions while the sensing NV dips are smeared across the frange by spatial
field variation, making the two components spectrally distinct.

The GF spectral shape is **the same ESR model as the sensing layer** — it
depends on the nitrogen isotope and crystal quality of the sensor diamond, not
the sample:

| Diamond type | GF shape |
|-------------|----------|
| 14N (standard) | Triple Lorentzian centred at ZFS (~2.8678, 2.870, 2.8722 GHz) |
| 15N | Doublet centred at ZFS |
| Degraded / mixed | Single broad peak at ZFS |

The implementation therefore uses the same `Model` instance as the fitting
pipeline, with the center parameter locked to ZFS:

```python
def estimate_fluorescence_correction_factor(
    data: ODMRData,          # mean-normalised
    model: Model,            # same model as used for fitting (ESR14N, ESR15N, ...)
    patch_size: int = 10,    # side length of patch around strongest pixel
) -> float:
    """Estimate GF correction factor from a two-component fit.

    Fits the spatial mean and a patch around the strongest pixel with:
      sensing_NV_envelope (free center) + GF_component (center locked to ZFS).
    The GF amplitude directly gives the correction factor.

    Only valid on mean-normalised data (not max-normalised).
    """
```

The patch-based approach reduces sensitivity to outlier bias: the strongest
single pixel may be an artifact; averaging a 10×10 neighbourhood (100 pixels)
gives a representative high-SNR spectrum from the high-field region while
suppressing noise and hot-pixel effects.

The correlation-minimisation sweep (Method D from the investigation) remains
a useful cross-check: sweep α over ~40 steps, apply correction, compute
`corr(off-resonance baseline, dip depth)` across all pixels, find the α
minimising `|corr|`.

---

## Limitations

| Limitation | Note |
|-----------|------|
| Auto-detection only works for spatially varying GF | GF uniform across all pixels is undetectable from normalised data alone |
| Removing max-norm is a breaking change | Any existing pipeline using `method='max'` will raise `ValueError`; migration: replace with `method='mean'` |
| Per-pixel amplitude equalisation can amplify noise | For pixels with near-zero dip depth, `pixel_range ≈ 0` → guard with a minimum threshold |
| Edge-baseline corruption | GF Lorentzian peaks sit at the frange edge — corrupting the baseline estimate; may need to exclude those frequency points from the edge baseline |
| Two-component fit requires the correct ESR model | GF shape matches the sensor diamond (14N triplet, 15N doublet, degraded single peak); use the same `Model` as the fitting pipeline with center locked to ZFS |

---

## Alternatives Considered

- **Single-pixel joint ESR+α fit**: Returns α=0 due to ESR parameter degeneracy on max-normalized data. Discarded.
- **Profile likelihood (per-pixel chi² sweep)**: Same degeneracy. Not useful without fixing normalization.
- **Keep max-norm as option**: Rejected — it is not physically valid for any pipeline that includes fluorescence correction. Leaving it available would create silent correctness errors.
- **Keep manual α**: Remains the fallback when auto-detection fails. Default should move from 0.2 → 0.5 after formula is aligned with MATLAB.
- **Apply correction before normalization**: Avoids the normalization issue but complicates the pipeline and diverges from the MATLAB reference approach.

---

## Implementation Plan

1. **Fix `NormalizationProcessor`**: remove `method='max'`; `method='mean'` becomes the only valid option; raise `ValueError` for `'max'` with migration message
2. **Fix `FluorescenceCorrectionProcessor`**: implement amplitude-equalisation term; update default `correction_factor` to 0.5
3. **Add `estimate_fluorescence_correction_factor`**: two-component fit (sensing NV envelope + GF component using same `Model` with center locked to ZFS) on global mean and 10×10 patch around strongest pixel; `correction_factor='auto'` mode; correlation-minimisation sweep as cross-check
4. **Tests**:
   - Unit: mean-norm produces correct normalisation on synthetic data
   - Unit: `ValueError` raised for `method='max'`
   - Unit: amplitude-equalisation correctness
   - Unit: auto-detection recovers known α on synthetic spatially-varying GF
   - Unit: patch extraction selects correct 10×10 neighbourhood around strongest pixel
5. **Notebook**: extend `fluorescence-correction-auto-alpha.ipynb` to validate the fixed pipeline and two-component fit
6. **Update CHANGELOG**
