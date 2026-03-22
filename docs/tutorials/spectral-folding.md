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

- **sqrt(2) noise reduction** in the folded spectrum (averages two measurements)
- **D_ZFS map** (temperature: dD/dT ~ -74 kHz/K; strain sensitivity)
- **Fold residual** — a model-free per-pixel quality metric
- **2x fewer fit calls** (one folded spectrum per polarity instead of two branches)

!!! warning "SNR trade-off"

    The sqrt(2) noise reduction applies to the raw folded spectrum. However,
    the D_ZFS estimation step introduces per-pixel errors (~0.05-0.15 MHz)
    that propagate into the fit. For **strong B111 signals** (std >> 2-3 uT),
    this error is negligible and folding gives a net accuracy benefit. For
    **weak signals** (B111 std < 2 uT), the D_ZFS error can dominate and the
    normal (unfolded) fit may produce more accurate B111 maps.

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

# Optional: exclude contaminated frequencies in folded (single-range) fit
# (folded fits accept the 'low' cutoff key)
result_cut = meas.fit_folded_odmr(freq_cutoff={'low': {'min': 2.8750}})

print(result.b111_remanent.shape)   # same shape as unfolded
```

The folded result is a drop-in replacement for the standard result: same
`b111_remanent`, `b111_induced`, and `magnetic_map` properties.

For deterministic testing or batch execution, `fit_folded_odmr()` also accepts
the same advanced overrides as normal fitting:

```python
result = meas.fit_folded_odmr(settings=my_settings, gpu_available=True)
```

Prefer built-in plotting helpers when a matching diagnostic exists:

```python
from qdmpy.plotting import (
    plot_b111_map,
    plot_folding_mean_spectrum,
    plot_folding_overview,
    plot_folding_pixel_spectra,
    plot_folding_search_landscape,
)

plot_b111_map(result.fit_result, component='remanent')
plot_folding_overview(folded)
plot_folding_search_landscape(folded)
plot_folding_mean_spectrum(folded)
plot_folding_pixel_spectra(folded, x=0, y=[0, 10, 140])
```

Use custom Matplotlib code only for plot types that do not yet have a dedicated
`qdmpy.plotting` helper.

---

## When to use folding

| Situation | Recommendation |
|-----------|----------------|
| Strong B111 signal (std >> 2-3 uT) | **Use folding** — noise averaging helps, D_ZFS error is negligible |
| Need D_ZFS / temperature map | **Use folding** — only folding gives D per pixel |
| Weak B111 signal (std < 2 uT) | **Skip folding** — D_ZFS estimation error dominates the noise benefit |
| Narrow ODMR linewidths (ESR15N, < 1 MHz) | **Caution** — D_ZFS estimation is harder with narrow dips; verify fold residual |
| Maximum B111 accuracy required | **Skip folding** — normal fit uses both branches independently |
| Faster processing needed | **Use folding** — halves the number of GPU fit calls |

---

## FoldingSettings parameters

Configure by creating a `FoldingSettings` instance and passing it to `fold_odmr()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_zfs_initial` | `2.870` | Starting center for D_ZFS search (GHz) |
| `search_range` | `0.005` | Half-width of D search window (GHz) |
| `search_steps` | `201` | Number of D candidates in the search grid |
| `bin_factor` | `8` | Spatial binning factor for coarse D_ZFS estimation |
| `interpolation_order` | `3` | Interpolation order for coarse->full D map (3 = bicubic) |
| `min_overlap_points` | `5` | Minimum overlap points required for valid folding |

Recommended ranges:

- `bin_factor`: 2-8 depending on scan SNR; higher = more robust D estimate but coarser
- `search_range`: ~0.002-0.010 GHz depending on expected D_ZFS variation
- `search_steps`: increase for finer D resolution at higher compute cost

Modify settings before calling `fold_odmr()`:

```python
from qdmpy.odmr.folding import FoldingSettings

folding_settings = FoldingSettings(
    bin_factor=8,
    search_range=0.006,
    search_steps=301,
)

folded = meas.fold_odmr(settings=folding_settings)
```

---

## Key takeaways

- Folding averages two measurements for sqrt(2) noise reduction per frequency
  point, but D_ZFS estimation error can offset this gain for weak signals
- **Strong signals** (B111 std >> 2-3 uT): folding helps — similar accuracy,
  2x fewer fit calls
- **Weak signals** (B111 std < 2 uT): normal fitting is more accurate
- Use folding when you need a D_ZFS map (temperature/strain) regardless of
  signal strength
- `meas.fold_odmr()` + `meas.fit_folded_odmr()` is the two-line quick path
- Use `freq_cutoff` on folded fits when near-center bins destabilize the fit
- Tune `FoldingSettings` when the default coarse D estimate is unreliable
- advanced callers can pass explicit `settings=` / `gpu_available=` to keep folded fits deterministic in tests and pipelines

---

## What's next

- [04 · Spectral Folding](04-spectral-folding.ipynb) — full interactive
  notebook with diagnostic plots and parameter sweeps
- [Fitting Quality](fitting.md) — interpreting chi2 and fit_states to assess
  whether folding improved your results
