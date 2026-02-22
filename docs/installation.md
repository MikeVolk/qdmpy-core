# Installation

qdmpy_core can be installed using pip or from source.

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## Installing with pip

The recommended way to install qdmpy_core is using pip:

```bash
pip install qdmpy_core
```

For development purposes, you can install qdmpy_core in editable mode:

```bash
git clone https://github.com/mikevolk/qdmpy_core.git
cd qdmpy_core
pip install -e .
```

## Using UV (recommended)

qdmpy_core can also be installed using UV, a fast Python package installer:

```bash
# Install UV if you don't have it
curl -sSf https://install.undefined.io/uv/ | python3 -

# Create a virtual environment and install qdmpy_core
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Dependencies

qdmpy_core has the following core dependencies:

- NumPy: For numerical operations
- Matplotlib: For plotting
- SciPy: For scientific computing

Optional dependencies include:

- pyGpufit: For GPU-accelerated fitting (highly recommended)

### Installing pyGpufit

For GPU-accelerated fitting, you'll need to install pyGpufit. This can be done via:

```bash
pip install pyGpufit
```

Or, for Windows users, you can use the pre-built wheel provided with qdmpy_core:

```bash
pip install src/pyGpufit/win/pyGpufit-1.2.0-py2.py3-none-any.whl
```

For Linux users:

```bash
pip install src/pyGpufit/linux/pyGpufit-1.2.0-py2.py3-none-any.whl
```

## Verifying Installation

After installation, you can verify that qdmpy_core is working correctly by importing it in Python:

```python
import qdmpy_core
print(qdmpy_core.__version__)
```