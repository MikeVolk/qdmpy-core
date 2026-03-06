# Quickstart

**Audience:** Fry &nbsp;|&nbsp; **Time:** ~5 min &nbsp;|&nbsp; **Prerequisites:** [Installation](installation.md)

---

## What you'll learn

- Load QDM data and fit ODMR spectra in three lines
- Access the B111 remanent and induced field maps
- Save and reload a result for later use
- Where to go next for each persona

---

## Setup

```python
import qdmpy
```

No additional imports needed for the common workflow.

---

## Load, fit, and inspect

### With real data

Point `qdmpy.load()` at the folder containing your `.mat` files from the QDM
microscope. Spatial binning and normalization are applied automatically.

```python
result = qdmpy.load('/data/FOV18x').fit_odmr()
```

That's it. `result` is a `QDMResult` containing all fitted parameters and
lazy access to field maps.

### Without data files

Use synthetic data for exploration or CI:

```python
result = qdmpy.make_synthetic_qdm_result(shape=(64, 64))
```

---

## B111 field maps

```python
b111_rem = result.b111_remanent   # (H, W) numpy array in µT
b111_ind = result.b111_induced    # (H, W) numpy array in µT
```

!!! note "Units"
    All B111 values are in **µT**. Frequencies (inside the fitting pipeline)
    are in **GHz**. These conventions are enforced throughout the library.

The remanent field isolates permanent magnetisation (ferro component); the
induced field tracks the applied bias (paramagnetic component).

### 3D field reconstruction

```python
mm = result.magnetic_map          # triggers Fourier-domain reconstruction
bz = mm.bz.values                 # xr.DataArray in µT
```

`magnetic_map` is computed lazily on first access and cached.

---

## Save and reload

```python
qdmpy.save_qdm(result, 'my_result.qdm')   # HDF5 format
result2 = qdmpy.load_qdm('my_result.qdm')
```

---

## Key takeaways

- `qdmpy.load(path).fit_odmr()` is the one-line entry point
- `result.b111_remanent` and `result.b111_induced` give 2D arrays in µT
- `result.magnetic_map` gives the full Bx/By/Bz reconstruction
- `save_qdm` / `load_qdm` for persistence

---

## What's next

- **Fry** — See [Notebook 01](tutorials/01-quickstart.ipynb) for a runnable
  version with plots and save/reload examples. Done.
- **Lila** — Continue to [Notebook 02: Exploring Data](tutorials/02-exploration.ipynb)
  to understand the processing pipeline and tune fit quality.
- **Professor** — Jump to [Notebook 03: Extending](tutorials/03-extending.ipynb)
  to build custom models, processors, and field reconstructors.
