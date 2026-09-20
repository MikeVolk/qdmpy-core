# QEP-ODMR-004 — Visualization Extraction

**Status:** Implemented
**Created:** 2026-02-22
**Revised:** 2026-03-22 — Marked Implemented; all spec items completed
**Severity:** LOW (L-5)
**Module:** `odmr/manager.py`

---

## Motivation

`ODMR.plot_spectra()` (lines 143-182) is a 40-line matplotlib method embedded
in a data lifecycle manager. It violates Single Responsibility — `ODMR` should
manage raw/processed data transitions, not rendering. The method imports
`matplotlib.pyplot` at call time, meaning any type-checker or static analyser
that follows imports sees a plotting dependency on the data layer.

## GUI Integration Requirements

No GUI impact. `ODMR.plot_spectra` is a matplotlib convenience method for
notebook/script use. The GUI uses pyqtgraph for its own spectrum rendering and
does not call `ODMR.plot_spectra` or import from `qdmpy.plotting`.

## Proposed Changes

### Move `plot_spectra` to `qdmpy/plotting/odmr.py`

```python
# qdmpy/plotting/odmr.py
def plot_odmr_spectra(odmr_data: ODMRData, y: int, x: int) -> None:
    """Plot all ODMR spectra for pixel (y, x) in a polarity x freq_range grid."""
    ...
```

Note: the final signature accepts `ODMRData` directly (not the `ODMR` manager),
which is a cleaner dependency — the plotting function depends on the data
container, not the lifecycle manager.

### Thin delegate on `ODMR`

```python
# odmr/manager.py
def plot_spectra(self, y: int, x: int, *, processed: bool = True) -> None:
    from qdmpy.plotting import plot_odmr_spectra
    odmr_data = self.processed_data if processed else self.raw_data
    plot_odmr_spectra(odmr_data, y, x)
```

This matches the delegate pattern used by `FitResult`, `MagneticMap`, and
`FoldedODMR` which all delegate to `qdmpy.plotting`.

## Migration

- Existing `odmr.plot_spectra(y, x)` calls continue to work via the delegate.
- Direct users of `from qdmpy.plotting import plot_odmr_spectra` gain a
  standalone function that accepts `ODMRData`.

## Test Plan

- [x] Smoke test: `plot_odmr_spectra` runs without error (`tests/test_plotting.py::TestODMRSpectraPlot::test_plot_odmr_spectra`)
- [x] Verify `ODMR.plot_spectra` delegates correctly (`tests/test_plotting.py::TestODMRSpectraPlot::test_odmr_manager_delegates`)
