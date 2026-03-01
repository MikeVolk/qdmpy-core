# Spectral Folding Tutorial

Spectral folding exploits the mirror symmetry of ODMR spectra to extract more
information from your data without additional measurements. The two frequency
ranges (low and high) are mirror images about the zero-field splitting D —
just like the two halves of a Mössbauer spectrum about the velocity zero.

[View the full tutorial notebook](../../notebooks/04-spectral-folding.ipynb)

## What you will learn

1. **The fold concept** — why ODMR spectra are symmetric about D_ZFS and what
   combining the two halves gives you
2. **Two-scale D_ZFS map** — estimating D per pixel at coarse spatial
   resolution then interpolating, to produce a temperature/strain map
3. **Per-pixel fold** — using the D_ZFS map to fold each pixel's spectrum,
   combining the two frequency ranges into a single higher-SNR spectrum
4. **Fitting** — passing the `FoldedODMR` to `fit_folded_odmr()` to obtain
   B111 maps with correct physics (centre IS the Zeeman offset δf)

## Background

For an NV center in a [111]-oriented diamond, the two ODMR resonance
frequencies are:

```
f± = D ± γ·B·cos(θ) + δ_strain
```

where D ~ 2.870 GHz is the zero-field splitting, γ = 28.024 GHz/T, and B is
the projected magnetic field. The low and high frequency ranges capture f− and
f+ respectively — symmetric about D. Folding them together gives:

- **√2 SNR improvement** for all fit parameters
- **D_ZFS map** (temperature: dD/dT ~ −74 kHz/K; strain)
- **Fold residual** — a model-free per-pixel quality metric
