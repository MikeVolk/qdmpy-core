# QEP-029 — FitManager: Separate Configuration from Execution

**Status:** Implemented (2026-02-20)
**Created:** 2026-02-18
**Supersedes (partially):** QEP-020 Fix 3 (data setter removal)

---

## Motivation

`FitManager.__init__` currently requires both configuration (model, constraints,
settings) *and* the data to be fitted:

```python
fm = FitManager(data, frequencies, model_name='ESR14N', constraints={...})
fm.fit_odmr()
params = fm.get_param('center')
```

This creates a structural tension that is already visible in the code:

- A `data` property setter exists (`fit.py:337`) so data can be replaced after
  construction — if data truly belonged at init, a setter would never be needed.
- `_reset_fit()` exists solely to invalidate derived state when data or
  configuration changes. It is called from the data setter, the `model_name`
  setter, and `set_constraints()`. This is defensive bookkeeping that only
  exists because configuration and data are coupled in the same object lifetime.
- `_fitted: bool` is a runtime flag rather than a type-level guarantee:
  `get_param()` and `parameter` raise `FitNotPerformedError` at runtime instead
  of the type system enforcing that a result exists.
- `fit_odmr()` is a void method that mutates five private arrays. Its results are
  only accessible afterwards via `get_param()` — a side-effect API rather than a
  value-returning one.

The natural boundary is: **construction = configuration, `fit()` = execution**.

---

## Goals

1. `FitManager.__init__` takes only stable configuration: model, constraints,
   settings, GPU flag.
2. `FitManager.fit(data, frequencies)` performs fitting and returns a `FitResult`.
3. No `data` setter, no `_reset_fit()`, no `_fitted` flag.
4. `FitManager` instances are reusable across different datasets with the same
   configuration (same model, constraints, hyperparameters).

---

## Design

### 3.1  `__init__` signature

```python
class FitManager:
    def __init__(
        self,
        model_name: str = 'ESR14N',
        constraints: dict[str, Any] | None = None,
        *,
        settings: QDMpySettings | None = None,
        gpu_available: bool | None = None,
    ) -> None:
```

`data` and `frequencies` are no longer accepted at construction. `ParameterGuesser`
receives frequencies at `fit()` time. `model_name='auto'` remains valid —
auto-detection is deferred to `fit()` where data is available.

### 3.2  `fit()` signature

```python
def fit(
    self,
    data: xr.DataArray,
    frequencies: NDArray,
) -> FitResult:
    """Fit ODMR spectra and return results.

    Args:
        data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).
        frequencies: Frequency array in GHz, shape (n_frange, n_freq).

    Returns:
        FitResult containing fitted parameters and quality metrics.
    """
```

From the caller's perspective `fit()` is pure: same inputs + configuration →
same result. It does not mutate `self`.

### 3.3  Auto model detection

When `model_name='auto'`, `fit()` calls `guess_model(flat_data)` with the
provided data. The resolved model is stored as `self._model` so that
`fit_manager.model_name` remains queryable after the call.

### 3.4  ParameterGuesser lifecycle

`ParameterGuesser` is constructed inside `fit()`, not in `__init__`. Its cache
is naturally scoped to one call and cache invalidation is not needed. If the
guess step proves expensive enough to warrant cross-call caching that can be
added as a later opt-in.

### 3.5  FitResult returned directly

`fit()` assembles a `FitResult` and returns it. The caller owns the result.
`FitManager` holds no accumulated state after the call.

### 3.6  Removed members

| Removed | Reason |
|---------|--------|
| `data` property + setter | Data no longer owned by FitManager |
| `_reset_fit()` | No mutable derived state to reset |
| `_fitted: bool` | Replaced by return-value semantics |
| `_fit_results`, `_states`, `_chi_squares`, `_number_iterations`, `_execution_time` | Returned via FitResult |
| `parameter` property | Results live on FitResult |
| `get_param()` | Results live on FitResult |
| `fitted` property | Replaced by return-value semantics |
| `fit_odmr()` | Renamed to `fit()` with return value |
| `_current_data_shape` | Scoped to `fit_frange()` call |

### 3.7  Kept members

| Kept | Notes |
|------|-------|
| `model_name` property + setter | Configuration; no data dependency |
| `set_constraints()` | Configuration |
| `set_free_constraints()` | Configuration |
| `constraints` property | Configuration |
| `get_constraints_array()` | Internal; called inside `fit()` |
| `get_constraint_types()` | Internal; called inside `fit()` |
| `fit_frange()` | Internal; kept as testable unit |
| `reshape_results()` | Internal |

### 3.8  Typical usage

```python
# Configure once
fm = FitManager(model_name='ESR14N', constraints={'width': {'vmax': 0.05}})

# Apply to data — returns a FitResult
result = fm.fit(data, frequencies)

# Reuse with different data — same model, same constraints
result2 = fm.fit(other_data, frequencies)
```

Update inside `Measurement.fit_odmr()` (`measurement.py:288`):

```python
# Current
fit_manager = FitManager(
    data=processed_data.data,
    frequencies=processed_data.frequencies,
    model_name=model_name,
    constraints=constraints,
)
fit_manager.fit_odmr()
parameters = self._extract_fit_parameters(fit_manager, model_name)

# After this QEP
fit_manager = FitManager(model_name=model_name, constraints=constraints)
result = fit_manager.fit(processed_data.data, processed_data.frequencies)
```

`_extract_fit_parameters` in `Measurement` becomes unnecessary since `FitResult`
already exposes the parameters directly.

---

## Interaction with QEP-020

QEP-020 Fix 3 proposes replacing the `data` setter with `update_data()`. This
QEP makes that moot: data is not stored on the instance at all.

QEP-020 Fixes 1 and 2 (flat_data cache, pre-allocated result arrays) are
complementary. Fix 1 becomes a local variable inside `fit()` rather than an
instance-level cache.

---

## Alternatives Considered

### A. Keep data in `__init__`, add a separate `refit(new_data)` method
Rejected. A reset mechanism and mutable state remain. Addresses the symptom, not
the design.

### B. Make FitManager a frozen dataclass
Rejected. `model_name` setter and `set_constraints()` are useful ergonomics for
interactive and notebook use. Immutability of *results* via return value is
sufficient.

### C. Return raw arrays from `fit()` instead of FitResult
Rejected. `Measurement.fit_odmr()` already constructs a `FitResult` — returning
it directly from `FitManager.fit()` removes an indirection layer and eliminates
`_extract_fit_parameters` entirely.

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/fit.py` | Remove data/frequencies from `__init__`; rename `fit_odmr()` → `fit()` returning `FitResult`; remove data setter, `_reset_fit()`, `_fitted`, accumulated result arrays |
| `src/QDMpy/measurement.py` | Update `fit_odmr()` to use new `FitManager.fit()` API; remove `_extract_fit_parameters` |
| `tests/test_fit.py` | Update all tests; add test for reuse across datasets |
