# QEP-005: Make Models Self-Describing (OCP Compliance)

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-003 (QEP-001 superseded by QEP-011) |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |

## Motivation

Adding a new ESR model currently requires changes in 4+ files:

1. `models.py` - define the model class and parameters
2. `fit.py` - add switch cases for parameter type identification
3. `measurement.py` - add `if model_name == "NewModel"` branch
4. `guess.py` - add model to peak-count mapping

This violates the Open/Closed Principle. The system should be open for extension
(new models) but closed for modification (existing code).

Specific OCP violations:

```python
# measurement.py:264-276
if model_name == "ESR14N":
    ...
elif model_name == "ESR15N":
    ...

# fit.py:451-464
if param_type == "center":
    ...
elif param_type == "contrast":
    ...
elif param_type == "width":
    ...
```


## Specification

### 1. Extend Model ABC with Self-Describing Properties

Add to the `Model` abstract base class:

```python
class Model(ABC):
    # ... existing interface ...

    @property
    @abstractmethod
    def parameter_types(self) -> dict[str, str]:
        """Map each parameter name to its type category.

        Returns:
            Dict mapping param name to one of:
            'center', 'width', 'contrast', 'offset'

        Example for ESR14N:
            {'center': 'center', 'width': 'width',
             'contrast_0': 'contrast', 'contrast_1': 'contrast',
             'contrast_2': 'contrast', 'offset': 'offset'}
        """

    @property
    @abstractmethod
    def frequency_parameters(self) -> list[str]:
        """Parameter names that are in frequency units.

        These parameters need GHz<->Hz conversion at the pygpufit boundary.
        Typically just ['center'].
        """

    @abstractmethod
    def default_constraints(self) -> dict[str, tuple[float, float, str]]:
        """Default fitting constraints for each parameter.

        Returns:
            Dict mapping param name to (lower_bound, upper_bound, constraint_type).
            constraint_type is one of: 'FREE', 'FIXED', 'LOWER', 'UPPER', 'LOWER_UPPER'
        """
```

### 2. Implement in Concrete Models

```python
class ESR14N(Model):
    @property
    def parameter_types(self) -> dict[str, str]:
        return {
            'center': 'center',
            'width': 'width',
            'contrast_0': 'contrast',
            'contrast_1': 'contrast',
            'contrast_2': 'contrast',
            'offset': 'offset',
        }

    @property
    def frequency_parameters(self) -> list[str]:
        return ['center']

    def default_constraints(self) -> dict[str, tuple[float, float, str]]:
        return {
            'center': (2.75, 2.95, 'LOWER_UPPER'),
            'width': (0.0001, 0.01, 'LOWER_UPPER'),
            'contrast_0': (0.0, 0.2, 'LOWER_UPPER'),
            'contrast_1': (0.0, 0.2, 'LOWER_UPPER'),
            'contrast_2': (0.0, 0.2, 'LOWER_UPPER'),
            'offset': (0.8, 1.2, 'LOWER_UPPER'),
        }
```

### 3. Eliminate Switch Statements

Replace `fit.py` switch on `param_type`:

```python
# Before (fit.py:451-464)
if param_type == "center":
    constraint[0] *= 1e9  # GHz to Hz
    constraint[1] *= 1e9
elif param_type == "contrast":
    ...

# After
for param_name in model.frequency_parameters:
    idx = model.param_names.index(param_name)
    constraints[idx] = self._convert_constraint_to_hz(constraints[idx])
```

Replace `measurement.py` switch on `model_name`:

```python
# Before
if model_name == "ESR14N":
    n_peaks = 3
elif model_name == "ESR15N":
    n_peaks = 2

# After
n_peaks = model.n_peaks  # Already defined on Model ABC
```

### 4. Simplify ModelRegistry

Change `ModelRegistry.register()` to accept a class reference:

```python
class ModelRegistry:
    _registry: dict[str, type[Model]] = {}

    @classmethod
    def register(cls, model_class: type[Model]) -> None:
        cls._registry[model_class.__name__] = model_class

    @classmethod
    def get(cls, name: str) -> type[Model]:
        return cls._registry[name]
```

Use decorator pattern for registration:

```python
@ModelRegistry.register
class ESR14N(Model):
    ...
```

## Files Affected

- `src/QDMpy/models.py` (extend ABC, implement in concrete models, simplify registry)
- `src/QDMpy/fit.py` (remove param_type switch, use model metadata)
- `src/QDMpy/measurement.py` (remove model_name switch, use model properties)
- `src/QDMpy/guess.py` (use `model.n_peaks` instead of iterating all models)
- `tests/test_models.py` (test new properties)
- `tests/test_fit.py` (update for new constraint flow)

## Backwards Compatibility

The `Model` ABC gains new abstract methods, which is a breaking change for any
external subclasses. Since no external models exist yet, this is acceptable.
The `ModelRegistry` API changes from dict-based to class-based registration.

## Verification

```bash
uv run pytest
uv run ruff check .
# Verify no model-name switch statements remain:
grep -rn "model_name ==" src/QDMpy/ | grep -v "__pycache__"  # Should be empty
grep -rn "param_type ==" src/QDMpy/ | grep -v "__pycache__"  # Should be empty
```

## Rejection Alternatives

**Alternative: Use a configuration dict instead of abstract methods.** Rejected
because abstract methods provide compile-time (type-check-time) guarantees that
all models implement the required interface. A dict can silently miss keys.

**Alternative: Use dataclasses for model metadata.** Considered but rejected as
unnecessary indirection. Properties on the model class are sufficient and keep
metadata co-located with the model logic.
