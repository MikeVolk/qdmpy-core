# QEP-046 — Notebook Tutorials for Three User Types

**Status:** Implemented
**Created:** 2026-02-21

---

## Motivation

The three user archetypes identified in the usability review have fundamentally
different entry points into QDMpy:

1. **Fit and be done** — load data, fit, get B111 maps, save output
2. **Exploratory** — inspect spectra, compare processors, tweak parameters,
   visualise results
3. **Developer** — add custom models or processors, extend the pipeline

Currently there is one `tutorial.ipynb` that partially covers User 1. Users 2
and 3 have no documented on-ramp. A developer who wants to add a model must
read source code.

---

## Goals

1. Three self-contained notebooks, one per user type.
2. Each notebook works with the provided synthetic data or the standard test
   data folder — no private data required to run them.
3. Notebooks are executable top-to-bottom with `uv run jupyter nbconvert
   --to notebook --execute notebooks/*.ipynb` in CI.
4. Each notebook demonstrates only what its target user needs — no information
   overload.

---

## Design

### Notebook 1 — `notebooks/01-quickstart.ipynb`

**Target**: User 1 — "I want to fit and be done"

**Contents**:
1. One-line load: `result = QDMpy.load('/data/FOV18x').fit_odmr()`
2. Access B111 remanent and induced maps
3. Save output as NetCDF
4. One overview plot

**Key API demonstrated**: `QDMpy.load()`, `QDMResult.b111_remanent`,
`QDMResult.show()`, `QDMResult.save()`

**Length**: ~10 cells, < 1 page equivalent

---

### Notebook 2 — `notebooks/02-exploration.ipynb`

**Target**: User 2 — "I want to play around with the data"

**Sections**:

1. **Loading and inspecting raw data**
   - `MatlabLoader`, `ODMRData` properties, shape inspection
   - `odmr.spectrum(y, x)` — plot a single spectrum

2. **Processing pipeline**
   - Add and chain processors: `BinningProcessor`, `NormalizationProcessor`,
     `OutlierProcessor`
   - Compare raw vs processed spectra side-by-side
   - `ODMR.plot_spectra(y, x)` — 2×2 grid

3. **Fitting**
   - `FitManager` directly (not via `Measurement`)
   - Compare `ESR14N` vs `auto` model
   - Inspect chi2 map and fit states

4. **Result exploration**
   - `result.plot('center')`, `result.plot('chi2')`
   - `result.b111` xr.Dataset — `.sel()`, `.plot()`
   - Access `magnetic_map.bz`, compare to `b111_remanent`

5. **Iteration**
   - Change bin factor, refit, compare results
   - Export to NetCDF and reload

**Key API demonstrated**: All of QEP-041 through QEP-044

---

### Notebook 3 — `notebooks/03-extending.ipynb`

**Target**: User 3 — "I want to develop my own algorithms"

**Sections**:

1. **Custom ESR model**
   - Subclass `Model`, implement `func()` and `guess()`
   - Register via `@ModelRegistry.register`
   - Use via `FitManager(model_name='MYMODEL')`
   - Verify against built-in model on synthetic data

2. **Custom processor**
   - Implement `Processor` protocol
   - Insert into `ODMRProcessorManager`
   - Before/after comparison

3. **Custom field reconstructor**
   - Implement `FieldReconstructor` protocol
   - Pass to `QDMResult(reconstructor=my_reconstructor)`
   - Compare output to default Fourier inversion

4. **Using `FitManager` standalone** (no `Measurement`)
   - Construct `FitManager`, call `.fit(data, freq)` directly
   - Wrap result in `QDMResult` manually

**Key API demonstrated**: `Model`, `Processor`, `FieldReconstructor`,
`ModelRegistry`, `FitManager`, `QDMResult`

---

## Synthetic data for notebooks

Notebooks 1 and 3 should not require a real data folder. Add a helper:

```python
# src/QDMpy/testing.py  (new small module)
def make_synthetic_measurement(
    shape: tuple[int, int] = (64, 64),
    bin_factor: int = 1,
    model: str = 'ESR14N',
    pixel_spacing: float = 4e-6,
) -> Measurement:
    """Create a synthetic Measurement for testing and tutorials."""
```

This is already partially available in `tests/conftest.py`; expose it publicly
so notebooks can use it without importing test infrastructure.

---

## CI Integration

Add to `pyproject.toml` or a `Makefile` target:

```bash
uv run jupyter nbconvert --to notebook --execute \
  notebooks/01-quickstart.ipynb \
  notebooks/02-exploration.ipynb \
  notebooks/03-extending.ipynb
```

Run this in CI on pushes to `claude` branch (not every commit, use path filter:
`notebooks/**` or `src/**`).

---

## Dependencies

| QEP | Why needed |
|-----|-----------|
| QEP-041 `QDMResult` | Notebooks 1 and 2 use `result.magnetic_map` |
| QEP-042 top-level exports | `import QDMpy; QDMpy.load()` in notebook 1 |
| QEP-043 `load()` entry point | Notebook 1 one-liner |
| QEP-044 convenience methods | Notebook 2 uses `odmr.spectrum()`, `result.plot()` |
| QEP-045 extension points | Notebook 3 uses `Model`, `Processor`, `FieldReconstructor` |

---

## Alternatives Considered

### A. Single comprehensive tutorial notebook
Rejected. Information overload. User 1 does not need to know about custom models.
Short focused notebooks are more likely to be read and kept up to date.

### B. Sphinx documentation with code examples
Complements but does not replace. Notebooks are runnable and interactive;
Sphinx docs are reference material. Both have value.

---

## Files to Create / Change

| File | Change |
|------|--------|
| `notebooks/01-quickstart.ipynb` | **New** |
| `notebooks/02-exploration.ipynb` | **New** |
| `notebooks/03-extending.ipynb` | **New** |
| `src/QDMpy/testing.py` | **New** — `make_synthetic_measurement()` |
| `.github/workflows/notebooks.yml` | **New** — CI notebook execution |
| `README.md` | Add links to all three notebooks under "Getting Started" |
