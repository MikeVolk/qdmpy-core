# QEP-020: Fix FitManager Memory Waste and Result Accumulation

**Status:** Superseded by QEP-029 (implemented 2026-02-20)
**Priority:** High
**Affects:** `fit.py`

## Problem

FitManager has two related performance/correctness issues that compound at
scale (typical data: 2 polarities x 2 frequency ranges x 4M pixels x 50 freqs
= ~3.2 GB).

### 1. `_flat_data` property copies data on every access

```python
@property
def _flat_data(self) -> NDArray:
    values = self._data_xr.values           # copy from xarray
    return values.reshape(n_pol, n_frange, -1, n_freq)  # may copy again
```

This property is called in:
- `__init__` → `guess_model(self._flat_data)` (line 262)
- `data` setter → `np.all(self._flat_data == data)` (line 339) — full array
  equality check
- `fit_odmr()` → `flat = self._flat_data` (line 614)
- `initial_parameter` → `self._guesser.guess(self._flat_data)` (line 525)
- `get_initial_parameter` → same (line 534)

For a 2k x 2k dataset with 2 polarities, 2 frequency ranges, and 50
frequencies, each `.values` call copies ~3.2 GB. During a typical `fit_odmr()`
call, `_flat_data` is accessed at least 3 times (init, guess, fit loop),
creating ~10 GB of unnecessary copies.

### 2. `fit_odmr()` result accumulation is fragile for >2 frequency ranges

```python
def fit_odmr(self, refit=False):
    for irange in range(flat.shape[1]):
        results = self.fit_frange(...)
        results = self.reshape_results(results)

        if self._fit_results is None:
            self._fit_results = results[0]        # first iteration: bare array
        else:
            self._fit_results = np.stack([self._fit_results, results[0]])
            # ^^ second iteration: (2, n_pol, n_pixel, n_params) — correct
            # third iteration: stack([shape(2,...), shape(n_pol,...)]) — WRONG
```

The `np.stack` approach works for exactly 2 frequency ranges but breaks for
any other count:
- **1 frequency range**: `np.swapaxes` at line 646 fails (no axis to swap).
- **3+ frequency ranges**: The third `np.stack` creates a nested structure
  `(2, ...)` stacked with `(n_pol, ...)`, producing a shape mismatch or
  wrong semantics.

The same pattern applies to `_states`, `_chi_squares`, `_number_iterations`,
and `_execution_time`.

### 3. `data` setter bypasses validation and is dangerous

```python
@data.setter
def data(self, data: NDArray) -> None:
    if np.all(self._flat_data == data):  # loads entire dataset for comparison
        return
    reshaped = data.reshape(n_pol, n_frange, n_y, n_x, n_freq)
    self._data_xr = xr.DataArray(reshaped, ...)
```

- The equality check `np.all(self._flat_data == data)` loads the full dataset
  into memory just to compare.
- The setter doesn't validate the new data (no shape checks, no NaN checks).
- It reconstructs xarray without preserving attributes or non-coordinate metadata.

## Proposed Fix

### Fix 1: Cache `_flat_data` and invalidate on data change

```python
def __init__(self, ...):
    self._flat_data_cache: NDArray | None = None

@property
def _flat_data(self) -> NDArray:
    if self._flat_data_cache is None:
        values = self._data_xr.values
        n_pol, n_frange = values.shape[0], values.shape[1]
        n_freq = values.shape[-1]
        self._flat_data_cache = values.reshape(n_pol, n_frange, -1, n_freq)
    return self._flat_data_cache

def _invalidate_cache(self):
    self._flat_data_cache = None
```

Call `_invalidate_cache()` from the data setter and `_reset_fit()`.

### Fix 2: Pre-allocate result arrays in `fit_odmr()`

```python
def fit_odmr(self, refit=False):
    flat = self._flat_data
    n_pol, n_frange, n_pixel, n_freq = flat.shape

    # Pre-allocate
    all_fit_results = np.empty((n_pol, n_frange, n_pixel, self.n_parameter), dtype=np.float32)
    all_chi_squares = np.empty((n_pol, n_frange, n_pixel), dtype=np.float32)
    all_states = np.empty((n_pol, n_frange, n_pixel), dtype=np.int32)
    execution_times = []

    for irange in range(n_frange):
        results = self.fit_frange(flat[:, irange], ...)
        results = self.reshape_results(results)
        all_fit_results[:, irange] = results[0]
        all_states[:, irange] = results[1]
        all_chi_squares[:, irange] = results[2]
        execution_times.append(results[4])

    self._fit_results = all_fit_results
    self._states = all_states
    self._chi_squares = all_chi_squares
    self._execution_time = np.array(execution_times)
    self._fitted = True
```

This:
- Works for any number of frequency ranges (1, 2, 3, ...).
- Eliminates the `np.stack` / `np.swapaxes` pattern.
- Pre-allocates memory instead of repeatedly creating new arrays.

### Fix 3: Remove or restrict the `data` setter

Either remove it entirely (data should be immutable after construction) or
replace with a method that validates:

```python
def update_data(self, data: xr.DataArray) -> None:
    """Replace data with a new validated DataArray."""
    self._validate_inputs(data, np.atleast_2d(self.f_ghz))
    self._data_xr = data
    self._invalidate_cache()
    self._guesser.reset()
    self._reset_fit()
```

## Impact

- **Memory**: Reduces peak memory from ~10 GB to ~3.2 GB for a typical dataset.
- **Correctness**: Fixes result shape for 1 or 3+ frequency ranges.
- **Safety**: Removes unvalidated data mutation path.

## Files to change

| File | Change |
|------|--------|
| `src/QDMpy/fit.py` | Cache _flat_data, pre-allocate results, fix/remove data setter |
| `tests/test_fit.py` | Add tests for 1 and 3 frequency ranges, cache invalidation |
