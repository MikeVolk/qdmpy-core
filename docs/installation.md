# Installation

QDMpy can be installed using pip or from source.

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## Installing with pip

The recommended way to install QDMpy is using pip:

```bash
pip install QDMpy
```

For development purposes, you can install QDMpy in editable mode:

```bash
git clone https://github.com/mikevolk/QDMpy.git
cd QDMpy
pip install -e .
```

## Using UV (recommended)

QDMpy can also be installed using UV, a fast Python package installer:

```bash
# Install UV if you don't have it
curl -sSf https://install.undefined.io/uv/ | python3 -

# Create a virtual environment and install QDMpy
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Dependencies

QDMpy has the following core dependencies:

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

Or, for Windows users, you can use the pre-built wheel provided with QDMpy:

```bash
pip install src/pyGpufit/win/pyGpufit-1.2.0-py2.py3-none-any.whl
```

For Linux users:

```bash
pip install src/pyGpufit/linux/pyGpufit-1.2.0-py2.py3-none-any.whl
```

## Verifying Installation

After installation, you can verify that QDMpy is working correctly by importing it in Python:

```python
import QDMpy
print(QDMpy.__version__)
```