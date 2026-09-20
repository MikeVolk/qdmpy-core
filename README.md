# QDMpy

> Python library for Quantum Diamond Microscopy (QDM) data analysis — load ODMR data, fit NV spectra, and generate quantitative magnetic field maps.

[![Tests](https://img.shields.io/github/actions/workflow/status/MikeVolk/qdmpy-core/test.yml?branch=main&label=tests&style=flat-square)](https://github.com/MikeVolk/qdmpy-core/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/licence-MIT-green?style=flat-square)](LICENCE)

**Docs:** [mikevolk.github.io/qdmpy-core](https://mikevolk.github.io/qdmpy-core) &nbsp;|&nbsp;
**Source:** [github.com/MikeVolk/qdmpy-core](https://github.com/MikeVolk/qdmpy-core) &nbsp;|&nbsp;
**Changelog:** [CHANGELOG.md](CHANGELOG.md)

> **Not on PyPI.** Install from source — see [Installation](#installation).

---

## Choose your path

| I want to... | Start here | Time |
|---|---|---|
| **Get B111 maps from my data as fast as possible** (Fry) | [Quickstart](docs/quickstart.md) → [Notebook 01](docs/tutorials/01-quickstart.ipynb) | 10 min |
| **Understand and tune my results** (Lila) | [Quickstart](docs/quickstart.md) → [Notebook 02](docs/tutorials/02-exploration.ipynb) → [Fitting Quality](docs/tutorials/fitting.md) | 1-2 h |
| **Build custom models, processors, or reconstructors** (Professor) | [Notebook 03](docs/tutorials/03-extending.ipynb) → [Source Fitting](docs/tutorials/06-source-fitting.ipynb) → [API Reference](docs/api/index.md) | open-ended |

---

## Installation

qdmpy-core is not published to PyPI. Install it from the repository.

**From a clone (recommended — required for the gpufit backend):**

```sh
git clone https://github.com/MikeVolk/qdmpy-core.git
cd qdmpy-core
uv venv && source .venv/bin/activate
uv pip install -e .
```

**Directly from git** (no clone; CPU/torch fitting only):

```sh
pip install git+https://github.com/MikeVolk/qdmpy-core.git
```

**Verify:**

```sh
python -c "import qdmpy; print(qdmpy.__version__)"
```

### Optional backends

| Extra | Install | Provides |
|---|---|---|
| `gpu` | `uv sync --extra gpu` | `torch` backend — CUDA, Apple-silicon MPS, and CPU |
| `gpufit` | `uv sync --extra gpufit` | `gpufit` backend — CUDA 11.5+ only, needs a clone |

`pyGpufit` is not on PyPI; the wheels are vendored under `src/pyGpufit/` and
resolved by uv from a checkout. Without either extra, fitting falls back to the
`scipy` backend, which needs no extra install.

See [Installation](docs/installation.md) for full details.

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

**Save and reload:**

```python
qdmpy.save_qdm(result, 'my_result.qdm')
result2 = qdmpy.load_qdm('my_result.qdm')
```

---

## Tutorials

| # | Notebook | Audience | Description |
|---|----------|----------|-------------|
| 01 | [01-quickstart.ipynb](docs/tutorials/01-quickstart.ipynb) | Fry | Load → fit → B111 maps → save |
| 02 | [02-exploration.ipynb](docs/tutorials/02-exploration.ipynb) | Lila | Pipeline, spectrum inspection, iteration |
| 03 | [03-extending.ipynb](docs/tutorials/03-extending.ipynb) | Professor | Custom model / processor / reconstructor |
| 04 | [04-spectral-folding.ipynb](docs/tutorials/04-spectral-folding.ipynb) | Lila/Professor | SNR improvement via spectral folding |
| 05 | [05-plotting.ipynb](docs/tutorials/05-plotting.ipynb) | All | Full plotting API walkthrough |
| 06 | [06-source-fitting.ipynb](docs/tutorials/06-source-fitting.ipynb) | Professor/Lila | Magnetic dipole source fitting |

---

## Development

### Setup

```sh
git clone https://github.com/MikeVolk/qdmpy-core.git
cd qdmpy-core
uv sync                       # project + `dev` group (ruff, ty, pytest, docs)
source .venv/bin/activate
pre-commit install
```

### Testing

```sh
uv run pytest
uv run pytest --cov=qdmpy --cov-report=term-missing
```

### Linting

```sh
pre-commit run --all-files
uv run ruff check .
uv run ty src/qdmpy
```

---

## License

MIT — see [LICENCE](LICENCE).
