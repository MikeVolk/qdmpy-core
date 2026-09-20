# QEP-045 — Developer Extension Points

**Status:** Implemented
**Created:** 2026-02-21

---

## Motivation

User 3 ("I want to develop my own algorithms") needs to:
- Add a custom ESR model
- Add a custom ODMR processor
- Swap the B111 → Bxyz reconstruction

The architecture already supports all three (registry pattern, protocol-based
processors, `MagneticMap` takes any B111 xr.DataArray). But the extension points
are invisible:

- `ModelRegistry` and `Model` are not exported from `QDMpy`
- `Processor` protocol is defined in `odmr/processors.py` but undocumented as a
  public interface
- There is no documented path to replace `MagneticMap` with a custom
  reconstructor
- No docstrings describe what a conforming implementation must provide

The result: users who read the source code can extend the library; users who
rely on the public API cannot.

---

## Goals

1. `Model`, `ModelRegistry`, and `Processor` are importable from `QDMpy`.
2. Each has a clear docstring describing the contract a custom implementation
   must satisfy.
3. A `FieldReconstructor` protocol is introduced so the magnetic reconstruction
   step is also pluggable.
4. `Measurement` (and `QDMResult` from QEP-041) accept a `reconstructor`
   parameter.

---

## Design

### 7.1  `Model` — custom ESR model

The `Model` ABC already exists in `fitting/models.py`. Required changes:

- Export from `QDMpy` (QEP-042 prerequisite)
- Add docstring to the ABC describing the required interface:

```python
class Model(ABC):
    """Base class for all ESR line-shape models.

    To add a new model, subclass Model and register it:

        @ModelRegistry.register
        class MyModel(Model):
            name: ClassVar[str] = 'MYRMODEL'
            n_peaks: ClassVar[int] = 1
            parameter_names: ClassVar[list[str]] = ['center', 'width', 'contrast', 'offset']
            frequency_parameters: ClassVar[list[str]] = ['center']

            def func(self, x: NDArray, params: NDArray) -> NDArray:
                # x shape: (n_freq,); params shape: (N, n_params)
                # return shape: (N, n_freq)
                ...

            def guess(self, x: NDArray, data: NDArray) -> NDArray:
                # x shape: (n_freq,); data shape: (N, n_freq)
                # return shape: (N, n_params) initial parameter guesses
                ...
    """
```

### 7.2  `Processor` — custom ODMR processor

`Processor` is currently a `typing.Protocol` in `odmr/processors.py`. Changes:

- Rename to `ODMRProcessor` (or keep `Processor` and re-export as both) to
  avoid collision with generic "processor" concepts
- Add docstring:

```python
class Processor(Protocol):
    """Protocol for ODMR data processors.

    Implement this protocol to add a custom processing step:

        class MyProcessor:
            def process(self, data: ODMRData) -> ODMRData:
                # Return a new ODMRData — never mutate the input.
                ...

            def describe(self) -> str:
                return 'MyProcessor(param=...)'

        odmr.processor_manager.add_processor(MyProcessor())
    """
    def process(self, data: ODMRData) -> ODMRData: ...
    def describe(self) -> str: ...
```

### 7.3  `FieldReconstructor` — pluggable magnetic reconstruction

New protocol in `src/QDMpy/magnetic.py` (or `src/QDMpy/protocols.py`):

```python
class FieldReconstructor(Protocol):
    """Protocol for B111 → 3D field reconstruction.

    Implement to replace the default Fourier inversion in MagneticMap:

        class MyReconstructor:
            def reconstruct(
                self,
                b111: xr.DataArray,   # dims (y, x), attrs['pixel_spacing'] in metres
                nv_axis: tuple[float, float, float],
            ) -> xr.Dataset:
                # Must return Dataset with variables: 'bx', 'by', 'bz', 'btotal'
                # Units: µT, dims (y, x) on each variable
                ...
    """
    def reconstruct(
        self,
        b111: xr.DataArray,
        nv_axis: tuple[float, float, float],
    ) -> xr.Dataset: ...
```

`MagneticMap.from_b111()` gains an optional `reconstructor: FieldReconstructor | None = None`
parameter. When `None`, the existing Fourier inversion is used.

`QDMResult` (QEP-041) gains `reconstructor: FieldReconstructor | None = None` and
passes it through to `MagneticMap.from_b111()`.

### 7.4  `ModelRegistry` — discover registered models

Export `ModelRegistry` and document listing:

```python
from QDMpy import ModelRegistry

# List all available models
ModelRegistry.available_models()   # ['ESR14N', 'ESR15N', 'ESRSINGLE', 'MYMODEL']

# Register a new model
@ModelRegistry.register
class MyModel(Model): ...

# Retrieve by name
model_cls = ModelRegistry.get('MYMODEL')
```

---

## Documentation deliverable

A `docs/extending.md` (or notebook `notebooks/03-developer-guide.ipynb`) that
walks through:
1. Adding a custom ESR model and registering it
2. Adding a custom processor and inserting it into the pipeline
3. Adding a custom field reconstructor

This is the entry point for User 3 and should be cross-referenced from the main
README.

---

## Alternatives Considered

### A. No `FieldReconstructor` protocol — just subclass `MagneticMap`
Rejected. `MagneticMap` is a frozen dataclass. Subclassing frozen dataclasses is
fragile. A protocol is more Pythonic and follows QDMpy's existing processor
pattern.

### B. Plugin system via entry_points (setuptools)
Deferred. Useful for third-party packages distributing models. Can be layered on
top of `ModelRegistry.register` later.

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/fitting/models.py` | Add contract docstring to `Model` ABC |
| `src/QDMpy/odmr/processors.py` | Add contract docstring to `Processor` protocol; export `Processor` |
| `src/QDMpy/magnetic.py` | Add `FieldReconstructor` protocol; add `reconstructor` param to `MagneticMap.from_b111()` |
| `src/QDMpy/result.py` | Add `reconstructor` param to `QDMResult` (QEP-041) |
| `src/QDMpy/__init__.py` | Export `Model`, `Processor`, `ModelRegistry`, `FieldReconstructor` (QEP-042) |
| `docs/extending.md` | **New** — developer guide |
| `tests/test_extensions.py` | **New** — tests for custom model, processor, reconstructor round-trips |
