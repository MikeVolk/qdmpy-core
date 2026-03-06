# Spectral Folding

**Audience:** Lila / Professor &nbsp;|&nbsp; **Time:** ~5 min read &nbsp;|&nbsp; **Prerequisites:** [Processors](processors.md)

For a full interactive walkthrough see [04 · Spectral Folding](04-spectral-folding.ipynb).

---

## What you'll learn

- Why spectral folding improves SNR and what you give up
- The quick two-line path: `fold_odmr()` + `fit_folded_odmr()`
- When folding helps vs when to skip it
- Key `FoldingSettings` parameters and their defaults

---

## Background

For an NV center in a [111]-oriented diamond, the two ODMR resonance
frequencies are:

```
f+/- = D +/- gamma * B * cos(theta) + delta_strain
```

where D ~ 2.870 GHz is the zero-field splitting, gamma = 28.024 GHz/T, and B
is the projected magnetic field. The low and high frequency ranges capture f-
and f+ respectively — symmetric about D. Folding them together gives:

- **sqrt(2) SNR improvement** for all fit parameters
- **D_ZFS map** (temperature: dD/dT ~ -74 kHz/K; strain sensitivity)
- **Fold residual** — a model-free per-pixel quality metric

---

## Quick path (Lila)

Two lines on top of a standard `Measurement`:

```python
import qdmpy

# Standard pipeline
meas = qdmpy.load('/data/FOV18x', bin_factor=2)

# 1. Fold the spectra (requires a prior coarse D_ZFS estimate)
folded = meas.fold_odmr()

# 2. Fit the folded spectra — returns QDMResult with improved SNR
result = meas.fit_folded_odmr()

print(result.b111_remanent.shape)   # same shape as unfolded
```

The folded result is a drop-in replacement for the standard result: same
`b111_remanent`, `b111_induced`, and `magnetic_map` properties.

---

## When to use folding

| Situation | Recommendation |
|-----------|----------------|
| Low SNR (chi2 >> 5, many unconverged pixels) | **Use folding** — sqrt(2) SNR gain helps |
| Need D_ZFS / temperature map | **Use folding** — only folding gives D per pixel |
| Maximum spatial resolution required | **Skip folding** — binning + folding both reduce effective resolution |
| Fast processing needed | **Skip folding** — folding adds a two-scale D estimation step |
| Standard 14N or 15N diamond | Either — try both and compare chi2 |

---

## FoldingSettings parameters

Accessible via `qdmpy.get_settings().folding`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coarse_bin_factor` | 4 | Spatial binning for the initial coarse D_ZFS estimate |
| `d_zfs_smoothing_sigma` | 2.0 | Gaussian smoothing radius (pixels) on the D_ZFS map before interpolation |
| `fold_threshold` | 0.005 | Maximum allowed residual after folding; pixels above threshold are flagged |
| `min_snr` | 3.0 | Minimum SNR to include a pixel in the folded fit |

Recommended ranges:

- `coarse_bin_factor`: 2–8 depending on scan SNR; higher = more robust D estimate but coarser
- `d_zfs_smoothing_sigma`: 1–5; increase for strain-uniform samples
- `fold_threshold`: 0.002–0.01; lower = stricter quality cut

Modify settings before calling `fold_odmr()`:

```python
from qdmpy import get_settings

settings = get_settings()
settings.folding.coarse_bin_factor = 8
settings.folding.d_zfs_smoothing_sigma = 3.0

folded = meas.fold_odmr()
```

---

## Key takeaways

- Folding provides sqrt(2) SNR improvement by exploiting ODMR mirror symmetry
- Use when chi2 is poor or you need D_ZFS maps; skip for maximum resolution
- `meas.fold_odmr()` + `meas.fit_folded_odmr()` is the two-line quick path
- Tune `FoldingSettings` when the default coarse D estimate is unreliable

---

## What's next

- [04 · Spectral Folding](04-spectral-folding.ipynb) — full interactive
  notebook with diagnostic plots and parameter sweeps
- [Fitting Quality](fitting.md) — interpreting chi2 and fit_states to assess
  whether folding improved your results
