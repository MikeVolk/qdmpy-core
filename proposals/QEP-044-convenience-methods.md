# QEP-044 — Convenience Methods for Exploration

**Status:** Implemented (2026-02-21)
**Created:** 2026-02-21

---

## Motivation

User 2 ("I want to play around with the data") is the most common QDM workflow: # note i assume this is not the most common (USer 1 will be)
inspect raw spectra, try processors, plot parameter maps, compare fits. The
current API makes each of these a multi-step operation:

**Inspect a spectrum at a pixel:**
```python
# Current — user must know the 5D array layout
spectrum = odmr.processed_data.data.sel(polarity='neg', freq_range='low')
                                    .values[y, x, :]
plt.plot(spectrum)
```

**Plot a parameter map:**
```python
# Current — user must import a separate plotting module
from QDMpy.plotting import plot_fit_result_parameter_map
plot_fit_result_parameter_map(result, 'center')
```

**Access B111 as an image:**
```python
# Current — two APIs for the same data, neither is self-documenting
result.b111_remanent          # NDArray shim
result.b111['remanent']       # xr.DataArray (different interface)
```

None of these are unreasonable individually, but together they create a steep
on-ramp for exploratory use.

---

## Goals

1. `odmr.spectrum(y, x, polarity='neg', freq_range='low')` — extract a single
   spectrum as a 1D array with its frequency axis.
2. `result.plot(param)` — thin wrapper that delegates to the plotting module.
3. `result.show()` — quick overview plot (wraps `plot_fit_result_overview`).
4. `QDMResult.plot(param)` and `QDMResult.show()` forwarded to `fit_result`.

---

## Design

### 6.1  `ODMR.spectrum()`

```python
def spectrum(
    self,
    y: int,
    x: int,
    polarity: str = 'neg',
    freq_range: str = 'low',
    *,
    processed: bool = True,
) -> tuple[NDArray, NDArray]:
    """Return (frequencies_ghz, intensities) for one pixel.

    Args:
        y, x: Pixel coordinates.
        polarity: 'neg' or 'pos'.
        freq_range: 'low' or 'high'.
        processed: If True, use processed_data; else raw_data.

    Returns:
        Tuple of (freq_ghz shape (n_freq,), intensity shape (n_freq,)).
    """
    data = self.processed_data if processed else self.raw_data
    freq = data.frequencies[0 if freq_range == 'low' else 1]   # (n_freq,)
    intensity = (data.data
                 .sel(polarity=polarity, freq_range=freq_range)
                 .values[y, x, :])
    return freq, intensity
```

Usage:
```python
freq, spec = odmr.spectrum(100, 200) # note this is only one side/polarity
plt.plot(freq, spec)
```

### 6.2  `FitResult.plot(param)`

```python
def plot(self, param: str = 'center', **kwargs) -> None:
    """Quick-plot a parameter map.

    Args:
        param: Parameter name ('center', 'linewidth', 'contrast', 'chi2', …).
        **kwargs: Forwarded to the underlying matplotlib imshow call.
    """
    from QDMpy.plotting import plot_fit_result_parameter_map
    plot_fit_result_parameter_map(self, param, **kwargs)
```

### 6.3  `FitResult.show()`

```python
def show(self, **kwargs) -> None:
    """Quick-plot overview of all fitted parameters and B111 maps."""
    from QDMpy.plotting import plot_fit_result_overview
    plot_fit_result_overview(self, **kwargs)
```

### 6.4  `QDMResult` delegation (see QEP-041)

`QDMResult.plot()` and `QDMResult.show()` delegate to `self.fit_result.plot()`
and `self.fit_result.show()` respectively. `QDMResult.magnetic_map` already
provides `.display()` for the magnetic components.

### 6.5  `ODMRData.plot_spectra(y, x)`

A convenience that plots both polarities and both freq ranges for a pixel in a
2×2 grid:

```python
def plot_spectra(self, y: int, x: int) -> None:
    """Plot all 4 ODMR spectra (2 polarities × 2 freq ranges) for pixel (y,x)."""
```

---

## Non-Goals

- Full interactive visualisation (widgets, sliders) — out of scope; belongs in
  a separate notebook/app layer.
- Replacing the plotting module — `plot_fit_result_*` functions remain the full
  API; these methods are thin wrappers.
- Adding `.plot()` to `MagneticMap` — it already has `.display()`.

---

## Alternatives Considered

### A. Jupyter widget-based inspector
Deferred. Adds a heavy dependency (ipywidgets). Can be built on top of this QEP.

### B. `__getitem__` on `ODMR` for spectrum access (`odmr[100, 200]`)
Rejected. `__getitem__` with a 2-tuple is non-obvious and not a Python idiom for
scientific data objects. A named method is clearer.

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/odmr/manager.py` | Add `ODMR.spectrum()`, `ODMR.plot_spectra()` |
| `src/QDMpy/odmr/data.py` | Add `ODMRData.plot_spectra()` (optional — may live only on `ODMR`) |
| `src/QDMpy/fitting/result.py` | Add `FitResult.plot()`, `FitResult.show()` |
| `src/QDMpy/result.py` | Add `QDMResult.plot()`, `QDMResult.show()` delegation |
| `tests/test_convenience.py` | **New** — tests for spectrum extraction, plot delegation (mock plt.show) |
