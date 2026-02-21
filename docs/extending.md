# Extending QDMpy

This guide covers the three main extension points for developers who want to
plug in custom algorithms without modifying the QDMpy source.

---

## 1. Custom ESR Model

### When to use this
Your NV centre has a different isotope, a strained crystal, or you want to test
an alternative line-shape (Gaussian, Voigt, etc.).

### Contract

Subclass `Model`, set the required class-level attributes, implement `func`,
and register with `@ModelRegistry.register`:

```python
from typing import ClassVar
import numpy as np
from numpy.typing import NDArray
from QDMpy import Model, ModelRegistry

@ModelRegistry.register
class MyModel(Model):
    name: ClassVar[str] = 'MYMODEL'

    def __init__(self) -> None:
        super().__init__(
            'MYMODEL',
            n_peaks=1,
            parameter_names=['center', 'width', 'contrast', 'offset'],
        )
        self.model_id = -1  # CPU-only; gpufit not used for custom models

    @property
    def parameter_types(self) -> dict[str, str]:
        return {
            'center': 'center',
            'width':  'width',
            'contrast': 'contrast',
            'offset': 'offset',
        }

    @property
    def frequency_parameters(self) -> list[str]:
        return ['center']   # parameters stored in GHz units

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        """Evaluate model spectrum.

        Args:
            x:          Frequency axis, shape (n_freq,), GHz.
            parameters: Parameter array, shape (N, n_params).

        Returns:
            Fluorescence array, shape (N, n_freq).
        """
        parameters = np.atleast_2d(parameters)
        center   = parameters[:, 0:1]
        width_sq = parameters[:, 1:2] ** 2
        contrast = parameters[:, 2:3]
        offset   = parameters[:, 3:4]
        dip = contrast * width_sq / ((x - center) ** 2 + width_sq)
        return 1 + offset - dip
```

### Using the model

```python
import QDMpy

m = QDMpy.load('/data/FOV18x')
result = m.fit_odmr(model='MYMODEL')
```

### Discovering registered models

```python
from QDMpy import ModelRegistry
ModelRegistry.available_models()
# ['ESR14N', 'ESR15N', 'ESRSINGLE', 'MYMODEL']
```

---

## 2. Custom ODMR Processor

### When to use this
You need a non-standard normalisation, a phase correction, or an
experiment-specific artefact removal step.

### Contract

Implement the `Processor` protocol — two methods, no base class required:

```python
from QDMpy.odmr.data import ODMRData

class MyProcessor:
    def process(self, data: ODMRData) -> ODMRData:
        """Return a **new** ODMRData — never mutate the input."""
        scaled = data.data * 1.05
        return ODMRData(data=scaled, metadata=data.metadata.copy())

    def describe(self) -> str:
        return 'MyProcessor(scale=1.05)'
```

### Inserting into the pipeline

```python
import QDMpy

m = QDMpy.load('/data/FOV18x')
m.odmr.processor_manager.add_processor(MyProcessor())
m.odmr.process_data()   # re-run pipeline with the new step
```

> **Immutability rule**: `process` receives an `ODMRData` and must return a
> *new* `ODMRData`. Mutating the input in place will corrupt the pipeline cache.

---

## 3. Custom Field Reconstructor

### When to use this
You have a custom inversion algorithm (e.g. iterative, noise-regularised,
or multi-plane) and want to replace the default Fourier method.

### Contract

Implement the `FieldReconstructor` protocol:

```python
import xarray as xr

class MyReconstructor:
    def reconstruct(
        self,
        b111: xr.DataArray,
        nv_axis: tuple[float, float, float],
    ) -> xr.Dataset:
        """Reconstruct Bxyz from B111.

        Args:
            b111:     DataArray (y, x), values in µT,
                      ``b111.attrs['pixel_spacing']`` in metres.
            nv_axis:  NV unit vector (ux, uy, uz).

        Returns:
            Dataset with variables 'bx', 'by', 'bz', 'btotal',
            each a DataArray (y, x) in µT.
        """
        zeros = xr.zeros_like(b111)
        return xr.Dataset({
            'bx': zeros,
            'by': zeros,
            'bz': b111,
            'btotal': abs(b111),
        })
```

### Using a custom reconstructor

Pass it to `QDMResult` or directly to `MagneticMap.from_b111()`:

```python
import QDMpy

result = QDMpy.load('/data/FOV18x').fit_odmr()

# Option A: via QDMResult (most common)
from QDMpy import QDMResult
qdm = QDMResult(fit_result=result.fit_result, reconstructor=MyReconstructor())
mm  = qdm.magnetic_map   # uses MyReconstructor

# Option B: directly
from QDMpy.magnetic_map import MagneticMap
import xarray as xr
b111_da = xr.DataArray(
    result.b111_remanent,
    dims=('y', 'x'),
    attrs={'pixel_spacing': result.pixel_spacing},
)
mm = MagneticMap.from_b111(b111_da, reconstructor=MyReconstructor())
```

---

## Protocol type-checking (optional)

All three protocols use `@runtime_checkable`, so you can verify compliance at
runtime:

```python
from QDMpy import Processor, FieldReconstructor
from QDMpy.fitting.models import Model

isinstance(MyProcessor(), Processor)               # True
isinstance(MyReconstructor(), FieldReconstructor)  # True
```
