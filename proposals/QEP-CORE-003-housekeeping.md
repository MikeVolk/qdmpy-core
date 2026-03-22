# QEP-CORE-003 — Low-Priority Housekeeping

**Status:** Draft
**Created:** 2026-02-22
**Severity:** LOW (L-1, L-4, L-6, L-8)
**Module:** Various

---

## Motivation

Four low-severity issues that individually don't warrant a QEP but
collectively improve code hygiene. None changes public API.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## L-1: `BlankSubtractor.blank` stores 2D array as nested tuple

**File:** `field_processing.py` (or equivalent field processing module)

`BlankSubtractor` stores its blank image as a nested `tuple[tuple[float, ...], ...]`
via Pydantic's `frozen=True` model. This forces `tuple(map(tuple, arr))` on
construction and `np.array(self.blank)` on every `process()` call — O(H*W)
conversion each way.

**Fix:** Use `NDArray` with `ConfigDict(arbitrary_types_allowed=True)` and
set `arr.flags.writeable = False` for immutability:

```python
class BlankSubtractor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    blank: NDArray

    @field_validator('blank')
    @classmethod
    def freeze_blank(cls, v: NDArray) -> NDArray:
        v = np.asarray(v)
        v.flags.writeable = False
        return v
```

---

## L-4: `Self` type annotation overuse

**Files:** Throughout codebase

Many methods annotate `self` as `self: Self` (from `typing.Self`). This is
only useful when the return type depends on `Self` (factory methods,
`__init__` returning the subclass type). For regular instance methods it adds
noise:

```python
# Unnecessary — Self adds nothing here
def get_parameter(self: Self, param_name: str) -> NDArray: ...

# Useful — Self ensures subclass return type
@classmethod
def load_results(cls: type[Self], filepath: Path) -> Self: ...
```

**Fix:** Remove `self: Self` annotations from methods that don't return
`Self`. Keep them on `@classmethod` constructors and methods that return
`self` for chaining.

This is a style-only change. Apply incrementally — no need to do it all at
once.

---

## L-6: `__init__.py` exports 30+ symbols

**File:** `src/QDMpy/__init__.py`

The top-level `__init__.py` imports and re-exports 30+ names (classes,
functions, constants). This has two effects:
1. `import QDMpy` triggers all module-level side effects (logging config,
   directory creation, model registration).
2. Tab-completion in IPython shows a wall of names, most of which users
   never need directly.

**Fix:** Define a minimal `__all__` with the ~10 most-used names:

```python
__all__ = [
    'Measurement',
    'ODMR',
    'ODMRData',
    'FitResult',
    'QDMResult',
    'FitManager',
    'MatlabLoader',
    'ModelRegistry',
    'load',
    'get_settings',
]
```

Other symbols remain importable via `from QDMpy.fitting.models import ESR14N`
but don't pollute the top-level namespace.

---

## L-8: `_make_params` in `testing.py` uses if/elif branching on model name

**File:** `src/QDMpy/testing.py`

```python
if model_name == 'ESR14N':
    params = ...
elif model_name == 'ESR15N':
    params = ...
elif model_name == 'ESRSINGLE':
    params = ...
```

This violates Open/Closed — adding a new model requires editing this function.

**Fix:** Use `ModelRegistry` to get parameter names dynamically:

```python
def _make_params(model_name: str, n_pol: int, n_frange: int,
                 n_pixels: int, rng: np.random.Generator) -> dict[str, NDArray]:
    model = ModelRegistry.get(model_name)
    params = {}
    for name in model.parameter_names:
        ptype = model.parameter_types[name]
        if ptype == 'center':
            params[name] = rng.normal(2.87, 0.001, (n_pol, n_frange, n_pixels))
        elif ptype == 'width':
            params[name] = rng.normal(0.0005, 1e-5, (n_pol, n_frange, n_pixels))
        elif ptype == 'contrast':
            params[name] = rng.uniform(0.01, 0.1, (n_pol, n_frange, n_pixels))
        elif ptype == 'offset':
            params[name] = rng.normal(0.0, 0.001, (n_pol, n_frange, n_pixels))
    params['chi2'] = rng.uniform(0, 0.01, (n_pol, n_frange, n_pixels))
    params['states'] = np.zeros((n_pol, n_frange, n_pixels), dtype=int)
    return params
```

Now any model registered in `ModelRegistry` works without code changes.

---

## Migration

All changes are internal. No public API impact.

## Test Plan

- [ ] L-1: Verify `BlankSubtractor` accepts NDArray, rejects mutation
- [ ] L-4: Verify `Self` removed from non-returning methods (grep check)
- [ ] L-6: Verify `__all__` contains expected names, tab-completion is clean
- [ ] L-8: Verify `_make_params('ESR14N')` still produces valid params
