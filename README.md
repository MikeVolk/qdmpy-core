# QDMpy

[![PyPI](https://img.shields.io/pypi/v/QDMpy?style=flat-square)](https://pypi.python.org/pypi/QDMpy/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/QDMpy?style=flat-square)](https://pypi.python.org/pypi/QDMpy/)
[![PyPI - License](https://img.shields.io/pypi/l/QDMpy?style=flat-square)](https://pypi.python.org/pypi/QDMpy/)
[![Tests](https://img.shields.io/github/actions/workflow/status/mikevolk/QDMpy/tests.yml?branch=master&label=tests&style=flat-square)](https://github.com/mikevolk/QDMpy/actions/workflows/tests.yml)

---

**Documentation**: [https://mikevolk.github.io/QDMpy](https://mikevolk.github.io/QDMpy)

**Source Code**: [https://github.com/mikevolk/QDMpy](https://github.com/mikevolk/QDMpy)

**PyPI**: [https://pypi.org/project/QDMpy/](https://pypi.org/project/QDMpy/)

---

## Overview

QDMpy is a comprehensive Python package for analyzing and visualizing Optically Detected Magnetic Resonance (ODMR) data from Quantum Diamond Microscopy (QDM) experiments. It provides tools for processing raw ODMR spectra, fitting spectral data to physics-based models, and generating quantitative magnetic field maps.

### Key Features

- **Data Processing Pipeline**: Customizable processing chain for ODMR data including normalization, binning, and outlier detection
- **Model-Based Fitting**: Automated and manual fitting of ODMR spectra with models for different nitrogen isotopes (14N, 15N)
- **Intuitive Visualization**: Comprehensive plotting functions for raw data, processed spectra, and magnetic field maps
- **Metadata Management**: Preservation of experimental parameters and processing history
- **Command-Line Interface**: Process QDM data without writing code
- **GPU-Accelerated Fitting**: Optimized performance through GPU computation (via pyGpufit)

## Installation

### Using uv (recommended)
```sh
uv pip install QDMpy
```

### Using pip
```sh
pip install QDMpy
```

### GPU Acceleration
GPU acceleration is automatically available when CUDA 11.5+ is installed on your system. The package includes bundled pyGpufit wheels for Windows and Linux platforms.

## Quick Start

```python
import QDMpy

# One-liner: load ODMR data, fit, get B111 field maps
result = QDMpy.load('/data/FOV18x').fit_odmr()

print(result.b111_remanent.shape)   # (H, W) numpy array, µT
print(result.b111_induced.shape)

result.save('my_result.npz')        # round-trip save/load
```

Try with synthetic data (no MATLAB files or GPU required):

```python
import QDMpy

result = QDMpy.make_synthetic_qdm_result(shape=(32, 32))
print(result.b111_remanent)

mm = result.magnetic_map            # full 3D reconstruction (Bx, By, Bz)
print(mm.bz.values)                 # xr.DataArray, µT
```

## Notebooks

| Notebook | Target user | Description |
|----------|-------------|-------------|
| [`notebooks/01-quickstart.ipynb`](notebooks/01-quickstart.ipynb) | Fit and be done | Load → fit → B111 maps → save |
| [`notebooks/02-exploration.ipynb`](notebooks/02-exploration.ipynb) | Exploratory | Pipeline, spectrum inspection, iteration |
| [`notebooks/03-extending.ipynb`](notebooks/03-extending.ipynb) | Developer | Custom model / processor / reconstructor |

## Command Line Usage

QDMpy includes a command-line interface for processing data without writing code:

```sh
# Process ODMR data with spatial binning
qdmpy process path/to/data --bin-factor 2 --model auto --output results

# Available options:
qdmpy process input_path \
  --output OUTPUT_DIR \
  --bin-factor 2 \
  --model {ESR14N,ESR15N,ESRSINGLE,auto} \
  --global-fluorescence 0.2 \
  --overwrite \
  --no-plots

# List available models
qdmpy models

# Get detailed model information
qdmpy models ESR15N --detailed

# Examine data file structure
qdmpy info path/to/data --summary
```

## Core Modules

- **odmr**: Complete ODMR data management and processing framework
  - `data`: ODMRData class for data encapsulation and metadata
  - `io`: Data loaders for MATLAB files and other formats
  - `processors`: Modular processing pipeline (binning, normalization, outlier detection)
  - `odmr`: Main ODMR orchestrator class
- **models**: Physics-based models with registry system (ESR14N, ESR15N, ESRSINGLE)
- **measurement**: Integration of ODMR data with optical reference images
- **fit**: GPU-accelerated fitting engine with constraint management
- **plotting**: Visualization tools for spectra and spatial maps
- **utils**: Utility functions for data processing and coordinate transformations
- **cli**: Command-line interface for batch processing workflows

## Development

### Requirements
- Python 3.12+
- uv (recommended) or pip

### Setup Development Environment

```sh
# Clone the repository
git clone https://github.com/mikevolk/QDMpy.git
cd QDMpy

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e .

# Verify GPU acceleration (if CUDA available)
uv run python -c "import QDMpy; print('GPU available:', QDMpy.PYGPUFIT_PRESENT)"
```

### Testing

```sh
# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=QDMpy --cov-report=term-missing

# Run single test
uv run pytest tests/test_file.py::test_function -v
```

### Linting and Type Checking

```sh
# Run all quality checks
pre-commit run --all-files

# Run individual checks
uv run ruff check .
uv run mypy src/QDMpy
```

## License

QDMpy is distributed under the [MIT License](LICENCE).

## Citation

If you use QDMpy in your research, please cite:

```
Volk, M. et al. (2023). QDMpy: A Python package for Quantum Diamond Microscopy data analysis.
Journal of Open Source Software, X(XX), XXXX. https://doi.org/10.XXXX/XXXX.XXXX
```

---

## Acknowledgments

QDMpy incorporates code and concepts from the quantum sensing community and relies on several open-source Python libraries. Special thanks to all contributors and the broader scientific community working on quantum sensing technologies.