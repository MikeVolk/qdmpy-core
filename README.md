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

```sh
pip install QDMpy
```

For GPU acceleration (recommended for large datasets):
```sh
pip install QDMpy[gpu]
```

## Quick Start

```python
from QDMpy import ODMR, Measurement
from QDMpy.odmr.io import MatlabLoader

# Load ODMR data from MATLAB files
loader = MatlabLoader(data_folder="path/to/data")
odmr_data = loader.load()

# Create ODMR instance and process data
odmr = ODMR(odmr_data)
odmr.process_data()  # Apply default processing pipeline

# Create a measurement with reference images
measurement = Measurement(
    odmr=odmr,
    light_image=light_img,
    laser_image=laser_img,
    output_directory="results"
)

# Fit ODMR spectra with appropriate model
from QDMpy.models import ModelRegistry
model = ModelRegistry.get("ESR14N")  # For 14N isotope
fit_parameters = measurement.fit_odmr(model)

# Generate and save magnetic field map
b_field = measurement.calculate_b_field()
measurement.plot_field_map(b_field, save=True)
```

## Command Line Usage

QDMpy includes a command-line interface for processing data without writing code:

```sh
qdmpy process path/to/data --binning 2 --model auto --output results
```

## Core Modules

- **odmr**: Management of ODMR spectral data and processing pipeline
- **models**: Physics-based models for fitting ODMR spectra
- **measurement**: Integration of ODMR data with optical images
- **io**: Data loading and saving from various file formats
- **plotting**: Visualization tools for QDM data analysis
- **utils**: Utility functions for data processing and manipulation

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
```

### Testing

```sh
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=QDMpy --cov-report=term-missing
```

### Linting and Type Checking

```sh
# Run all quality checks
pre-commit run --all-files

# Run individual checks
ruff check .
mypy src/QDMpy
```

## License

QDMpy is distributed under the [MIT License](LICENSE).

## Citation

If you use QDMpy in your research, please cite:

```
Volk, M. et al. (2023). QDMpy: A Python package for Quantum Diamond Microscopy data analysis.
Journal of Open Source Software, X(XX), XXXX. https://doi.org/10.XXXX/XXXX.XXXX
```

---

## Acknowledgments

QDMpy incorporates code and concepts from the quantum sensing community and relies on several open-source Python libraries. Special thanks to all contributors and the broader scientific community working on quantum sensing technologies.