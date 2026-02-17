# QEP-014: Fix Type Safety Errors in Core Package

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | None |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-16 |

## Motivation

The project uses `ty` for type checking. Running `ty check` on the core package
(excluding `plotting.py`, deferred to QEP-012) reports **15 errors in 6 files**.
These aren't cosmetic — several indicate real bugs where the code calls methods
or attributes that don't exist on the current API.

### Real bugs (will crash at runtime)

| File | ty Rule | Description |
|------|---------|-------------|
| `fit.py:638` | `unresolved-attribute` | `Model` has no attribute `model_id` — GPU fitting broken |
| `cli/qdmpy_cli.py:322` | `unresolved-attribute` | `MatlabLoader` has no attribute `get_file_list` — CLI broken |
| `cli/calculate_QDMio.py:22` | `unresolved-import` | Imports `QDMpy._core.qdm_old` which doesn't exist |
| `cli/qdmpy_cli.py:225` | `unresolved-import` | Same dead import of `QDMpy._core.qdm_old` |
| `models.py:353,374` | `missing-argument` | `Model()` called without required args — registry crashes |
| `guess.py:104` | `missing-argument` | Same `Model()` without args issue |

### Type narrowing / stub issues (not runtime bugs but must be fixed)

| File | ty Rule | Description |
|------|---------|-------------|
| `fit.py:582-585` | `no-matching-overload` | `np.stack` with `NDArray \| None` — ty can't narrow past the if/else |
| `settings.py:174` | `invalid-method-override` | `settings_customise_sources` signature incompatible with pydantic supertype |
| `odmr/processors.py:90` | `unresolved-attribute` | `DataArrayCoarsen.mean` — xarray stubs incomplete |
| `guess.py:154,181,211` | `not-iterable` | numba `prange` not recognized as iterable by ty |

### Impact

- **GPU fitting path is broken** — `model_id` doesn't exist, `fit_frange()` crashes
- **CLI is broken** — both `process` and `calculate_QDMio` import a deleted module (`_core.qdm_old`)
- **Model registry has a typing hole** — `Model()` without args works at runtime only because concrete subclasses override `__init__`, but ty correctly flags the base class signature mismatch

## Specification

### 1. Remove dead `_core.qdm_old` imports in CLI

**Root cause:** `cli/calculate_QDMio.py` and `cli/qdmpy_cli.py` import from
`QDMpy._core.qdm_old`, a module that was deleted during the refactoring.

**Fix:** Rewrite these CLI commands to use the current API (`Measurement`,
`ODMR`, `FitManager`), or remove the commands entirely if they're
superseded. The `calculate_QDMio.py` script is likely legacy and should be
deleted or rewritten.

### 2. Fix `Model` constructor / registry pattern

**Root cause:** `Model.__init__` requires `name`, `n_peaks`, `parameters_unique`
as positional arguments, but `ModelRegistry.register()` calls `model_cls()` with
no arguments, and `ModelRegistry.get()` calls `cls._registry[name]()` with no
arguments.

**Fix:** Make the concrete model subclasses (`ESR14N`, `ESR15N`, `ESRSINGLE`)
define their own `__init__` with hardcoded defaults, so they can be instantiated
without arguments:

```python
class ESR14N(Model):
    def __init__(self) -> None:
        super().__init__(
            name='ESR14N',
            n_peaks=3,
            parameters_unique=['center', 'width', 'contrast_0',
                               'contrast_1', 'contrast_2', 'offset'],
        )
```

Alternatively, if the subclasses already do this, the issue may be that ty
sees the base class signature. In that case, add an `__init__` override to the
base `Model` class to accept optional arguments, or use `@overload`.

### 3. Add `model_id` property to `Model`

**Root cause:** `fit.py:638` references `self._model.model_id` but Model has
no such attribute. This is the pygpufit model identifier (an integer enum).

**Fix:** Add `model_id` as an abstract property on `Model`:

```python
class Model(ABC):
    @property
    @abstractmethod
    def model_id(self) -> int:
        """pygpufit model ID for GPU fitting."""
```

Each concrete model implements it with the correct pygpufit constant:

```python
class ESR14N(Model):
    @property
    def model_id(self) -> int:
        import pygpufit.gpufit as gf
        return gf.ModelID.ESR_14N
```

If pygpufit isn't installed, this property should either:
- Raise `ImportError` (acceptable — only called when GPU fitting)
- Return a sentinel constant defined locally

### 4. Fix `np.stack` with potentially-None arrays

**Root cause:** `fit.py:581-585` does:
```python
self._fit_results = np.stack((self._fit_results, results[0]))
```
But `self._fit_results` is typed as `NDArray | None` and could be None.

**Fix:** Restructure to collect results in a list and stack once at the end:

```python
all_results = []
for irange in range(n_franges):
    results = self.fit_frange(...)
    all_results.append(self.reshape_results(results))

self._fit_results = np.stack([r[0] for r in all_results])
self._states = np.stack([r[1] for r in all_results])
# etc.
```

The list-then-stack approach is cleaner and eliminates the None handling entirely.

### 5. Fix `settings_customise_sources` signature

**Root cause:** The pydantic-settings `BaseSettings` method signature uses
`PydanticBaseSettingsSource` as the parameter type, but our override uses more
specific subtypes (`InitSettingsSource`, `EnvSettingsSource`, etc.) and adds
`**kwargs`. ty flags this as an LSP violation.

**Fix:** Match the superclass signature exactly, using the more general types:

```python
@classmethod
def settings_customise_sources(
    cls,
    settings_cls: type[BaseSettings],
    init_settings: PydanticBaseSettingsSource,
    env_settings: PydanticBaseSettingsSource,
    dotenv_settings: PydanticBaseSettingsSource,
    file_secret_settings: PydanticBaseSettingsSource,
    **kwargs: Any,
) -> tuple[PydanticBaseSettingsSource, ...]:
```

### 6. Fix `MatlabLoader.get_file_list` in CLI

**Root cause:** `qdmpy_cli.py:322` calls `MatlabLoader.get_file_list()` but
this method doesn't exist on the current `MatlabLoader` class.

**Fix:** Either:
- Add `get_file_list()` to `MatlabLoader` / `BaseLoader`
- Replace the CLI call with the equivalent operation from the current API
  (likely `glob` or `Path.iterdir()`)

### 7. Suppress false positives with targeted comments

For issues where ty is wrong (incomplete stubs or numba-specific constructs):

```python
# xarray stubs don't expose .mean() on DataArrayCoarsen
result = data.coarsen(y=bin_factor, x=bin_factor).mean()  # type: ignore[unresolved-attribute]

# numba.prange is iterable at runtime (JIT-compiled)
for i in prange(n):  # type: ignore[not-iterable]
```

## Files Affected

- `src/QDMpy/models.py` — fix constructor, add `model_id`
- `src/QDMpy/fit.py` — fix None stacking, use `model_id`
- `src/QDMpy/guess.py` — fix Model instantiation, suppress `prange`
- `src/QDMpy/settings.py` — fix override signature
- `src/QDMpy/cli/qdmpy_cli.py` — fix dead import, fix `get_file_list`
- `src/QDMpy/cli/calculate_QDMio.py` — rewrite or remove dead import
- `src/QDMpy/odmr/processors.py` — suppress xarray stub limitation
- `tests/test_models.py` — test `model_id` property
- `tests/test_fit.py` — test stacking refactor

## Verification

```bash
# Zero ty errors in core (excluding plotting.py)
uv run ty check src/QDMpy/ 2>&1 | grep -v plotting.py | grep "^error"
# Expected: no output

# All existing tests still pass
uv run pytest -x -q
# Expected: 287 passed, N skipped

# Ruff clean
uv run ruff check src/QDMpy/ --exclude src/QDMpy/plotting.py
# Expected: only TRY003 errors remain
```

## Rejection Alternatives

**Alternative: Suppress all ty errors with `type: ignore`.** Rejected — the
`model_id`, `get_file_list`, `_core.qdm_old`, and `Model()` errors are real bugs
that will crash at runtime. Suppressing them hides real defects.

**Alternative: Switch back to mypy.** Not relevant — ty catches the same real
bugs plus additional issues (numba prange, dead imports). The tool choice
doesn't change the underlying problems.

**Alternative: Fix incrementally.** All 15 core errors are straightforward
fixes. No reason to defer — fix them all in one pass.
