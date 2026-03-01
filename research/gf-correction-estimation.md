# GF Correction Factor Estimation — Research Notes

**Date:** 2026-02-25
**Dataset:** `tests/data/MIL2_FOV1` (2 pol, 2 frange, 1200×1920 px, binned 4× → 300×480)
**Expected α:** ~0.2 (from prior manual tuning)

---

## Background

The MATLAB correction formula (from `correct_global.m`):

```
corrected(i,j,k) = (baselinerange / pixelrange) * (data(i,j,k) - meanbaseline)
                   - globalFraction * specZBL(k) + globalmeanBL
```

- **Op 1 (amplitude equalization):** normalizes per-pixel dip depth to match the global
- **Op 2:** subtracts `α * spec_zbl` (the zero-baselined global mean spectrum)

The global signal = `mean(data, [1 2])` — spatial mean at each frequency, per (pol, frange) slice. Confirmed identical in MATLAB and Python implementations.

---

## Approach 1: Correlation Minimization (FAILED)

Sweep α ∈ [0, 0.8], apply correction, compute `corr(baseline, dip_depth)` across all pixels, find α minimizing |corr|.

**Result:** Did not produce reliable estimates. See QEP-048 for full investigation.

---

## Approach 2: Cluster Fitting (current investigation)

**Idea:** Find the strongest NV pixels (5×5 cluster), model GF as spectrally flat, fit the global signal and normal NV signal.

### Pixel selection

Tested two methods:
- **Max dip depth** (baseline − min): selected (2, 152) — edge pixel, likely artifact
- **Max MAD** (mean absolute deviation from spatial mean): selected (48, 113) — more central, better

**Recommendation:** Use MAD with edge margin (5px) for pixel selection.

### What works

Fit `S_global(f) = a + b · S_cluster(f)` where S_cluster is the 5×5 cluster mean.

`1-b` represents the fraction of the global spectrum NOT explained by the cluster shape — a proxy for GF contamination.

**Results:**

| pol | frange | 1-b | cluster dip offset |
|-----|--------|-----|--------------------|
| 0 | 0 | 0.276 | -5 indices |
| 0 | 1 | 0.227 | +5 indices |
| 1 | 0 | 0.550 | +8 indices |
| 1 | 1 | 0.561 | -8 indices |

### What doesn't work

**Per-pixel fitting within the cluster:** fitting each cluster pixel as `S(f) = a + b·template(f)` gives `a ≈ baseline ≈ 1.0`. Subtracting cluster_bl yields `a - cluster_bl ≈ 0` because all cluster pixels are too similar to each other. Cannot isolate GF this way.

**3-component amp_eq fit:** fitting `amp_eq(f) = α·spec_zbl(f) + b·cluster_mean(f) + c` gives α ≈ 0 with high std (~0.1–0.2) — spec_zbl and cluster_mean are collinear.

**Dip depth ratio:** `global_range / cluster_range ≈ 0.93–1.0` — too close to 1 to be useful.

---

## Analysis: Shape Mismatch Problem

The `1-b` estimate conflates two effects:
1. **GF dilution** (what we want) — the global mean has GF that the cluster doesn't
2. **Spectral shape mismatch** — the cluster has sharp dips at field-specific positions, the global mean has smeared dips from averaging over all field values

### Evidence

The global mean dip is always at **index 24** (center of 51-point spectrum) — the smeared average. The cluster dips are offset:
- pol=0: ±5 indices → cluster shape at global dip position ≈ 91–94% of max depth
- pol=1: ±8 indices → cluster shape at global dip position ≈ 79–84% of max depth

The 60% larger offset for pol=1 explains why `1-b` is ~0.55 (shape mismatch dominates) vs ~0.25 for pol=0 (shape mismatch is smaller, GF signal dominates).

### Physics

In QDM, the sample magnetic field adds to the bias field for one polarity and subtracts for the other. The cluster is in a strong-field region (high MAD). For the polarity where sample field opposes bias, the sensing dips are pushed back toward ZFS. But the *rest* of the image has dips scattered over different positions. When averaged, the global dip sits at the center, far from the cluster's specific position.

The GF itself (bulk NVs at zero field) produces peaks pinned at ZFS, always near the frange boundary. This is the same for both polarities — GF doesn't depend on the applied field.

---

## Open Questions

1. **Can we correct for shape mismatch?** e.g., align dip positions before fitting, or use a model-based approach that separates dip position from amplitude
2. **Should we average only pol=0?** It has smaller dip offset and gives `1-b ≈ 0.25`, closer to expected 0.2. But this feels fragile.
3. **Multiple clusters:** Using several clusters (e.g., top-5 MAD pixels, each with 5×5) could average out field-dependent biases
4. **Two-component model fit:** Fit `S_global = α·L(f; ZFS) + (1-α)·NV_envelope(f)` where L is a Lorentzian at ZFS (the known GF shape) — avoids the shape mismatch problem entirely
5. **Use raw (unnormalized) data?** Mean-normalization compresses the baseline differences. The GF signal might be more detectable in raw data.

---

## Summary

The cluster-fitting approach (`S_global = a + b·S_cluster`) gives reasonable α estimates for pol=0 (~0.25) but is contaminated by spectral shape mismatch for pol=1 (~0.55). The mismatch arises because the cluster's dip positions are field-dependent and don't align with the global smeared mean.

Next step: either correct for the mismatch, use only the less-affected polarity, or adopt a model-based approach that explicitly models the GF spectral shape (Lorentzian at ZFS).
