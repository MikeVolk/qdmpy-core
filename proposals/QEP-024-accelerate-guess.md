# QEP-024: Accelerate Initial Parameter Guessing

**Status:** Implemented (commit ec75819, 2026-02-20)
**Priority:** High
**Affects:** `guess.py`, `fit.py` (ParameterGuesser)

---

## Problem

`guess.py` is one of the two main bottlenecks in the pipeline (alongside
`pygpufit`-based fitting). For a typical 2k × 2k scan the code processes
~4 million pixels across 2 polarities and 2 frequency ranges, calling
per-pixel Numba JIT functions millions of times.

### Root causes

#### 1 — Redundant cumsum computation

`normalize_pixel` computes a cumulative sum over the frequency axis. It is
called inside both `guess_center_pixel` **and** `guess_width_pixel`, so for
every pixel the cumsum is computed **twice**. For 4 M pixels this is 8 M
independent cumsum operations instead of 4 M.

#### 2 — Scalar Numba is the wrong tool

The `@njit(parallel=True)` approach processes one pixel at a time in the inner
body:

```python
for px in prange(n_pixel):
    centers[p, r, px] = guess_center_pixel(data[p, r, px, :], freq[r])
```

Each call to `guess_center_pixel` is a small scalar kernel that allocates
temporary arrays (`normalize_pixel` returns a new array), then extracts a
single float. At 4 M pixels this creates 4 M short-lived temporary allocations.
Numba's parallel executor threads are doing real work, but the overhead
swamps the useful computation for operations that are natively vectorized in
NumPy.

#### 3 — JIT compilation cold-start

First use of any `@njit` function incurs a compilation penalty (~0.5–2 s per
function). Even with Numba's cache this adds latency on cold imports and
complicates debugging.

#### 4 — Only the innermost loop is parallelized

The outer loops over `n_pol` and `n_frange` are plain Python `range` loops.
For the typical (2, 2, 4M, 50) array this leaves the Numba thread pool idle
for most of the iteration tree.

---

## Proposed Solution: Two-Phase Approach

### Phase 1 — Vectorized NumPy (remove Numba entirely)

Replace all `@njit` functions with fully vectorized NumPy operations over the
full 4D array. NumPy's internal BLAS/SIMD path is designed exactly for this
workload.

#### Normalized cumsum — computed once, shared

```python
def _normalize_all(data: NDArray) -> NDArray:
    """Compute normalized cumulative sum for all pixels simultaneously.

    Args:
        data: (n_pol, n_frange, n_pixel, n_freq)

    Returns:
        (n_pol, n_frange, n_pixel, n_freq) in [0, 1]
    """
    cs = np.cumsum(data - 1.0, axis=-1)
    mn = cs.min(axis=-1, keepdims=True)
    rng = cs.max(axis=-1, keepdims=True) - mn
    safe = np.where(rng > 0, rng, 1.0)
    return (cs - mn) / safe
```

#### Contrast — trivially vectorized

```python
def guess_contrast(data: NDArray) -> NDArray:
    mx = np.nanmax(data, axis=-1)
    mn = np.nanmin(data, axis=-1)
    with np.errstate(invalid='ignore'):
        result = np.abs((mx - mn) / np.where(mx != 0, mx, 1.0))
    return np.where(mx != 0, result, 0.0)
```

#### Center — argmin over normalized cumsum

```python
def guess_center(data: NDArray, freq: NDArray) -> NDArray:
    # freq: (n_frange, n_freq)
    normalized = _normalize_all(data)                         # (pol, fr, px, f)
    idx = np.argmin(np.abs(normalized - 0.5), axis=-1)        # (pol, fr, px)
    # index into freq per freq-range with advanced indexing
    frange_idx = np.arange(data.shape[1])[:, np.newaxis]      # (fr, 1)
    return freq[frange_idx, idx]                               # (pol, fr, px)
```

#### Width — two argmins over same normalized cumsum

```python
def guess_width(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> NDArray:
    normalized = _normalize_all(data)
    lidx = np.argmin(np.abs(normalized - vmin), axis=-1)
    ridx = np.argmin(np.abs(normalized - vmax), axis=-1)
    frange_idx = np.arange(data.shape[1])[:, np.newaxis]
    return np.abs(freq[frange_idx, ridx] - freq[frange_idx, lidx])
```

#### Shared normalization in `guess_initial_fit_parameters`

Compute `_normalize_all` once inside `guess_initial_fit_parameters` and pass
the result to both `guess_center` and `guess_width`. This eliminates the
redundant cumsum entirely.

```python
def guess_initial_fit_parameters(
    data: NDArray, freq: NDArray, model: Model
) -> NDArray:
    normalized = _normalize_all(data)   # single cumsum pass

    parameter_guessers = {
        "center":   lambda: _guess_center_from_normalized(normalized, freq),
        "contrast": lambda: guess_contrast(data),
        "width":    lambda: _guess_width_from_normalized(normalized, freq, DEFAULT_VMIN, DEFAULT_VMAX),
        "offset":   lambda: np.ones(data.shape[:3]),
    }
    ...
```

### Phase 2 — Optional GPU backend via CuPy (opt-in)

`pygpufit` already runs the fitting on the GPU. The guessing step immediately
precedes it but runs entirely on CPU. Moving guessing to the GPU eliminates a
CPU→GPU transfer that would otherwise be followed immediately by GPU fitting.

**Strategy**: thin dispatch layer that tries `cupy`, falls back to `numpy`.

```python
try:
    import cupy as xp
    _BACKEND = 'cupy'
except ImportError:
    import numpy as xp
    _BACKEND = 'numpy'
```

All vectorized functions from Phase 1 use `xp` instead of `np` — the API is
identical. No logic changes required.

**Data flow with CuPy enabled:**

```
ODMRData (CPU numpy)
  └─ to_cupy()                 # H2D transfer, ~40 MB for 4M×50 array
  └─ _normalize_all()          # GPU
  └─ guess_center/width/contrast  # GPU
  └─ initial_params (cupy)     # stays on GPU
  └─ pygpufit (already GPU)    # zero-copy handoff via DLPack / pointer
```

**Dependency**: `cupy-cuda12x` (or matching CUDA version). Added as an
optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
gpu = ["cupy-cuda12x>=13.0"]
```

Install with `uv pip install -e ".[gpu]"`. Falls back transparently when
absent.

---

## Alternatives Considered

| Approach | Verdict |
|----------|---------|
| Improve existing Numba (parallelize outer loops, share cumsum) | Reduces some overhead but Numba per-pixel dispatch is still slower than vectorized NumPy for this workload. More complex to maintain. |
| PyTorch | Heavier dependency, different API from NumPy, overkill for element-wise ops. |
| JAX | Functional style requires architectural changes (no in-place ops), adds compilation overhead, difficult on non-Linux. |
| C extension / Cython | Maintenance burden with no clear advantage over vectorized NumPy. |
| Numba CUDA kernels | Possible but requires writing GPU kernels manually. CuPy gives the same benefit with zero extra code. |

---

## Expected Performance

| Config | Estimated time (2k × 2k, ESR14N) |
|--------|----------------------------------|
| Current (Numba parallel) | ~3–8 s |
| Phase 1 — vectorized NumPy | ~0.2–0.8 s |
| Phase 2 — CuPy GPU | ~0.02–0.1 s |

The main gains come from:
- Eliminating per-pixel function-call overhead (~10–50×)
- Eliminating redundant cumsum (2×)
- GPU bandwidth (additional 5–10×)

---

## Migration Plan

1. **Phase 1** — vectorize, no external deps:
   - Replace all `@njit` functions in `guess.py` with vectorized NumPy.
   - Keep public API identical (`guess_contrast`, `guess_center`, `guess_width`,
     `guess_initial_fit_parameters`).
   - Remove `numba` from `dependencies` in `pyproject.toml` only if it is not
     used elsewhere; otherwise keep but remove from `guess.py` imports.
   - Add parametrized regression tests comparing output of old vs new for
     random inputs.
   - Benchmark both implementations; record results in PR.

2. **Phase 2** — CuPy backend (separate PR, after Phase 1 lands):
   - Add `gpu` optional extra.
   - Implement `_get_array_module()` utility.
   - Thread `xp` through all guessing functions.
   - Add `use_gpu: bool` setting to `QDMpySettings` (default `False`).
   - Test on systems with and without CUDA.

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/guess.py` | Rewrite all `@njit` functions as vectorized NumPy; add shared `_normalize_all` helper; expose `xp` dispatch hook for Phase 2 |
| `src/QDMpy/settings.py` | Add `use_gpu: bool = False` to `QDMpySettings` (Phase 2) |
| `pyproject.toml` | Add `[gpu]` optional extra for CuPy (Phase 2); possibly remove `numba` dep |
| `tests/test_guess.py` | Regression tests: old vs new output; benchmark fixture |
| `CHANGELOG.md` | Entry under Performance |
