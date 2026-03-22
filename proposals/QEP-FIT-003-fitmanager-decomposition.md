# QEP-FIT-003 — FitManager.fit() Decomposition

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-5)
**Module:** `fitting/manager.py`

---

## Motivation

`FitManager.fit()` (lines 149-241) is a ~90-line method that handles input
validation, auto-model resolution, data flattening, parameter guessing,
per-frange GPU dispatch, result transposition, parameter dict construction,
quality metrics, metadata assembly, and `FitResult` construction. It is the
most important method in the codebase and the hardest to test because each
step is entangled with the others.

Additionally, `_param_idx()` (lines 394-412) contains undocumented legacy
aliases (`"resonance"` -> `"center"`, `"mean_contrast"` -> `"contrast"`) that
create a hidden API surface.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Proposed Changes

### 1. Extract pipeline stages as private methods

```python
def fit(self, data: xr.DataArray, frequencies: NDArray, *,
        pixel_spacing: float = 1.0) -> FitResult:
    self._check_gpu()
    self._validate_inputs(data, frequencies)

    flat, f_ghz, spatial = self._prepare_data(data, frequencies)
    model = self._ensure_model(flat)
    initial = self._guess_parameters(model, flat, f_ghz)
    raw_results = self._fit_all_franges(flat, f_ghz, initial, model)
    return self._assemble_result(raw_results, model, spatial, pixel_spacing)
```

### 2. Each helper is independently testable

| Method | Responsibility | Lines |
|--------|---------------|-------|
| `_check_gpu()` | Raise DependencyError if no GPU | 3 |
| `_prepare_data(data, freq)` | Flatten to 4D, extract dimensions | ~10 |
| `_ensure_model(flat)` | Resolve auto or return existing | ~5 |
| `_guess_parameters(model, flat, freq)` | Create guesser, get initial params | ~5 |
| `_fit_all_franges(flat, freq, initial, model)` | Loop over franges, collect raw results | ~20 |
| `_assemble_result(raw, model, spatial, px)` | Transpose, build param dict, metrics, FitResult | ~25 |

### 3. Remove undocumented aliases from `_param_idx`

The `"resonance"` -> `"center"` and `"mean_contrast"` -> `"contrast"` aliases
are undocumented and untested. Remove them. If backward compatibility is
needed, add explicit deprecation warnings for one release.

## Migration

- No public API change — `fit()` signature and return type are unchanged.
- Internal test patches may change if tests mock intermediate steps.
- `_param_idx` alias removal may break code that uses `"resonance"` or
  `"mean_contrast"` strings.

## Test Plan

- [ ] Unit test each extracted helper independently
- [ ] Integration test: full `fit()` still returns identical `FitResult`
- [ ] Verify `_param_idx` raises `ParameterError` for removed aliases
- [ ] Verify no performance regression from method call overhead
