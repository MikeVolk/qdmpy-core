# QDMpy

> Python library for Quantum Diamond Microscopy (QDM) data analysis — load ODMR data, fit NV spectra, and generate quantitative magnetic field maps.

[![Tests](https://img.shields.io/github/actions/workflow/status/MikeVolk/qdmpy-core/test.yml?branch=main&label=tests&style=flat-square)](https://github.com/MikeVolk/qdmpy-core/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/licence-MIT-green?style=flat-square)](https://github.com/MikeVolk/qdmpy-core/blob/main/LICENCE)

---

## Choose your path

| I want to... | Start here | Time |
|---|---|---|
| **Get B111 maps from my data as fast as possible** (Fry) | [Quickstart](quickstart.md) → [Notebook 01](tutorials/01-quickstart.ipynb) | 10 min |
| **Understand and tune my results** (Lila) | [Quickstart](quickstart.md) → [Notebook 02](tutorials/02-exploration.ipynb) → [Fitting Quality](tutorials/fitting.md) | 1-2 h |
| **Build custom models, processors, or reconstructors** (Professor) | [Notebook 03](tutorials/03-extending.ipynb) → [Source Fitting](tutorials/06-source-fitting.ipynb) → [API Reference](api/index.md) | open-ended |

---

## Three lines to B111 maps

```python
import qdmpy

result = qdmpy.load('/data/FOV18x').fit_odmr()

print(result.b111_remanent)   # (H, W) ndarray in µT
print(result.b111_induced)    # (H, W) ndarray in µT
```

**No data files?** Use synthetic data (CI-friendly, no GPU required):

```python
import qdmpy

result = qdmpy.make_synthetic_qdm_result(shape=(64, 64))
print(result.b111_remanent.shape)   # (64, 64)

mm = result.magnetic_map            # Fourier-domain 3D reconstruction
print(mm.bz.values)                 # xr.DataArray in µT
```

---

## Installation

qdmpy-core is not published to PyPI — install it from the repository:

```sh
git clone https://github.com/MikeVolk/qdmpy-core.git
cd qdmpy-core
uv venv && source .venv/bin/activate
uv pip install -e .
```

See [Installation](installation.md) for GPU fitting setup and full details.

---

## Tutorials

| # | Notebook | Audience | Description |
|---|----------|----------|-------------|
| 01 | [01-quickstart.ipynb](tutorials/01-quickstart.ipynb) | Fry | Load → fit → B111 maps → save |
| 02 | [02-exploration.ipynb](tutorials/02-exploration.ipynb) | Lila | Pipeline, spectrum inspection, iteration |
| 03 | [03-extending.ipynb](tutorials/03-extending.ipynb) | Professor | Custom model / processor / reconstructor |
| 04 | [04-spectral-folding.ipynb](tutorials/04-spectral-folding.ipynb) | Lila/Professor | SNR improvement via spectral folding |
| 05 | [05-plotting.ipynb](tutorials/05-plotting.ipynb) | All | Full plotting API walkthrough |
| 06 | [06-source-fitting.ipynb](tutorials/06-source-fitting.ipynb) | Professor/Lila | Magnetic dipole source fitting |

See the [Tutorial Overview](tutorials/index.md) for persona-based navigation.

---

## Key Links

- [Migration Guide](migration.md) — upgrading from older QDMpy versions
- [Changelog](changelog.md) — release history
- [API Reference](api/index.md) — full public API documentation
- [Extending QDMpy](extending.md) — protocols for custom algorithms
