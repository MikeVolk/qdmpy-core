# Installation

**Prerequisites:** Python 3.12 or higher.

---

## Standard install

=== "uv (recommended)"

    ```sh
    uv pip install qdmpy-core
    ```

=== "pip"

    ```sh
    pip install qdmpy-core
    ```

**Verify:**

```sh
python -c "import qdmpy; print(qdmpy.__version__)"
```

---

## GPU fitting (optional)

CPU fitting works out of the box. For large datasets (>500k pixels), GPU
acceleration via [pyGpufit](https://pypi.org/project/pyGpufit/) significantly
reduces fit time.

**Requirements:** CUDA 11.5+ and a compatible NVIDIA GPU.

```sh
pip install pyGpufit
```

Check availability at runtime:

```python
import qdmpy
print(qdmpy.is_pygpufit_available())   # True if GPU fitting is ready
```

If `pyGpufit` is not installed or CUDA is unavailable, qdmpy automatically
falls back to CPU fitting with no code changes required.

---

## Development install

```sh
git clone https://github.com/mikevolk/QDMpy.git
cd QDMpy
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
pre-commit install
```

Run the test suite to verify the install:

```sh
uv run pytest
```
