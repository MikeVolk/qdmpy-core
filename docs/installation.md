# Installation

**Prerequisites:** Python 3.13 or higher, and [uv](https://docs.astral.sh/uv/).

qdmpy-core is **not published to PyPI**. Install it from the repository.

---

## Standard install

=== "From a clone (recommended)"

    ```sh
    git clone https://github.com/MikeVolk/qdmpy-core.git
    cd qdmpy-core
    uv venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    uv pip install -e .
    ```

=== "Directly from git"

    ```sh
    pip install git+https://github.com/MikeVolk/qdmpy-core.git
    ```

    This gives you the `scipy` backend. The `gpufit` backend needs a clone
    (see below), because its wheels are vendored in the repository.

**Verify:**

```sh
python -c "import qdmpy; print(qdmpy.__version__)"
```

---

## GPU fitting (optional)

CPU fitting via the `scipy` backend works out of the box. For large datasets
(>500k pixels), a GPU backend significantly reduces fit time. Two are
available.

### torch backend (portable)

Runs on CUDA, Apple-silicon MPS, and CPU. Installs from PyPI:

```sh
uv sync --extra gpu
```

```python
result = measurement.fit_odmr(backend='torch')
```

### gpufit backend (CUDA only)

Fastest for large 14N/15N fits, but **requires CUDA 11.5+ and a clone of this
repository**. pyGpufit is not distributed on PyPI; the wheels are vendored
under `src/pyGpufit/` and resolved by uv from the checkout:

```sh
uv sync --extra gpufit
```

Check availability at runtime:

```python
import qdmpy
print(qdmpy.is_pygpufit_available())   # True if GPU fitting is ready
```

If neither extra is installed, `backend='auto'` selects the `scipy` backend
with no code changes required.

---

## Development install

```sh
git clone https://github.com/MikeVolk/qdmpy-core.git
cd qdmpy-core
uv sync            # project + the `dev` dependency group (ruff, ty, pytest, docs)
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pre-commit install
```

`dev` is a [dependency group](https://peps.python.org/pep-0735/), not an extra,
so `uv sync` picks it up automatically — `uv pip install -e ".[dev]"` does not.

Run the test suite to verify the install:

```sh
uv run pytest
```
