# QEP-006: Proper Dependency Injection

| Field | Value |
|-------|-------|
| **Status** | Finished |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-002 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |

## Motivation

`FitManager` and `ModelRegistry` import the global `SETTINGS` singleton directly,
creating tight coupling that makes unit testing require heavy mocking:

```python
# fit.py:21
from QDMpy import PYGPUFIT_PRESENT, SETTINGS

# models.py:23
from QDMpy import SETTINGS
```

This violates the Dependency Inversion Principle: high-level modules (`FitManager`)
depend on low-level details (global `SETTINGS` object) instead of abstractions.

Consequences:
- Unit tests must mock module-level imports or patch globals
- Cannot test GPU and CPU code paths without mocking import-time state
- Cannot test with different settings without global mutation
- Import order matters (circular import risk)


## Specification

### 1. FitManager Accepts Optional Settings

```python
class FitManager:
    def __init__(
        self,
        frequencies: np.ndarray,
        data: np.ndarray,
        model_name: str,
        *,
        settings: QDMpySettings | None = None,
        gpu_available: bool | None = None,
    ):
        self._settings = settings or get_settings()
        self._gpu_available = (
            gpu_available if gpu_available is not None
            else is_pygpufit_available()
        )
```

This enables:
```python
# Production code (unchanged behavior)
fm = FitManager(freq, data, "ESR14N")

# Test code (no mocking needed)
fm = FitManager(freq, data, "ESR14N", settings=test_settings, gpu_available=False)
```

### 2. GPU Path Testing

With `gpu_available` as a constructor parameter, both code paths can be tested:

```python
def test_fit_cpu_path():
    fm = FitManager(..., gpu_available=False)
    result = fm.fit_frange(...)
    assert result.shape == expected_shape

def test_fit_gpu_path():
    fm = FitManager(..., gpu_available=True)
    # Test GPU path (will use actual GPU if available, or skip)
```

### 3. Move Default Constraints to Models

Per QEP-005, models define their own default constraints. `FitManager` no longer
needs to read constraint defaults from settings. Instead:

```python
# Before (FitManager reads settings for constraints)
constraints = getattr(self._settings, f'{model_name}_constraints')

# After (FitManager asks model for defaults, settings can override)
constraints = self._model.default_constraints()
if self._settings.has_constraint_overrides(model_name):
    constraints = self._settings.get_constraint_overrides(model_name)
```

### 4. ConstraintManager Injection

`ConstraintManager` should receive its configuration through the constructor:

```python
class ConstraintManager:
    def __init__(
        self,
        model: Model,
        *,
        constraint_overrides: dict | None = None,
    ):
        self._model = model
        self._constraints = model.default_constraints()
        if constraint_overrides:
            self._constraints.update(constraint_overrides)
```

## Files Affected

- `src/QDMpy/fit.py` (add DI parameters to `FitManager`, `ConstraintManager`)
- `src/QDMpy/models.py` (remove direct `SETTINGS` import)
- `src/QDMpy/measurement.py` (pass settings to `FitManager`)
- `tests/test_fit.py` (simplify by using DI instead of mocking)
- `tests/test_measurement.py` (simplify mocking)

## Backwards Compatibility

All new parameters are optional with defaults matching current behavior:
- `settings=None` defaults to `get_settings()`
- `gpu_available=None` defaults to `is_pygpufit_available()`

Existing code calling `FitManager(freq, data, model)` continues to work unchanged.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run mypy src/QDMpy
# Verify no direct SETTINGS imports in fit.py or models.py:
grep -n "from QDMpy import.*SETTINGS" src/QDMpy/fit.py src/QDMpy/models.py
# Should show zero matches
```

## Rejection Alternatives

**Alternative: Use a DI container (dependency-injector, inject).** Rejected as
over-engineering for a codebase with ~5 injectable dependencies. Constructor
injection is sufficient and requires no additional dependencies.

**Alternative: Use environment variables for test configuration.** Rejected
because it's fragile (global state via a different mechanism) and doesn't solve
the GPU path testing problem.
