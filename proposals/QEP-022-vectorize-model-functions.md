# QEP-022: Vectorize Model Evaluation Functions

**Status:** Implemented
**Priority:** Medium
**Affects:** `models.py`

## Problem

The three model evaluation functions (`esr14n`, `esr15n`, `esrsingle`) use
Python `for` loops to iterate over parameter rows:

```python
def esr14n(x, parameter, ahyp=AHYP_14N):
    out = []
    parameter = np.atleast_2d(parameter)
    for p in parameter:           # Python loop over N spectra
        aux1 = x - p[0] + ahyp
        width_squared = p[1] * p[1]
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)
        ...
        out.append(1 + p[5] - dip1 - dip2 - dip3)
    return np.array(out)          # Builds list, then converts
```

### Performance impact

These functions are called during:
1. **Model evaluation for visualization** — evaluating the model at every pixel
   to compare against data.
2. **Initial guess validation** — checking if guessed parameters produce
   reasonable spectra.
3. **Residual computation** — calculating fit quality.

For 4 million pixels (2k x 2k), the Python loop iterates 4M times. Each
iteration creates temporary arrays (`aux1`, `aux2`, `aux3`, `dip1`, etc.) and
appends to a list. This is orders of magnitude slower than vectorized numpy.

### The fix is trivial

The Lorentzian formula is naturally vectorized. With broadcasting, the same
computation handles all spectra simultaneously:

```python
def esr14n(x, parameter, ahyp=AHYP_14N):
    parameter = np.atleast_2d(parameter)
    center = parameter[:, 0:1]     # (N, 1) for broadcasting
    width = parameter[:, 1:2]
    c0, c1, c2 = parameter[:, 2:3], parameter[:, 3:4], parameter[:, 4:5]
    offset = parameter[:, 5:6]

    width_sq = width * width
    dip1 = c0 * width_sq / ((x - center + ahyp)**2 + width_sq)
    dip2 = c1 * width_sq / ((x - center)**2 + width_sq)
    dip3 = c2 * width_sq / ((x - center - ahyp)**2 + width_sq)

    return 1 + offset - dip1 - dip2 - dip3
```

This processes all N spectra in a single vectorized operation.

**Note**: These Python model functions mirror the GPU kernels and are primarily
used for non-GPU evaluation (visualization, testing, CPU fallback). The GPU
kernels are already parallelized by pyGpufit. But for any CPU-side evaluation,
vectorization provides 100-1000x speedup.

## Proposed Fix

Vectorize all three model functions:

### `esr14n` (14N, 3 dips, 6 parameters)

```python
def esr14n(x, parameter, ahyp=AHYP_14N):
    parameter = np.atleast_2d(parameter)
    x = np.atleast_1d(x)
    c, w = parameter[:, 0:1], parameter[:, 1:2]
    w_sq = w * w
    return (
        1 + parameter[:, 5:6]
        - parameter[:, 2:3] * w_sq / ((x - c + ahyp)**2 + w_sq)
        - parameter[:, 3:4] * w_sq / ((x - c)**2 + w_sq)
        - parameter[:, 4:5] * w_sq / ((x - c - ahyp)**2 + w_sq)
    )
```

### `esr15n` (15N, 2 dips, 5 parameters)

```python
def esr15n(x, parameter, ahyp=AHYP_15N):
    parameter = np.atleast_2d(parameter)
    x = np.atleast_1d(x)
    c, w = parameter[:, 0:1], parameter[:, 1:2]
    w_sq = w * w
    return (
        1 + parameter[:, 4:5]
        - parameter[:, 2:3] * w_sq / ((x - c + ahyp)**2 + w_sq)
        - parameter[:, 3:4] * w_sq / ((x - c - ahyp)**2 + w_sq)
    )
```

### `esrsingle` (1 dip, 4 parameters)

```python
def esrsingle(x, parameter):
    parameter = np.atleast_2d(parameter)
    x = np.atleast_1d(x)
    c, w = parameter[:, 0:1], parameter[:, 1:2]
    w_sq = w * w
    return (
        1 + parameter[:, 3:4]
        - parameter[:, 2:3] * w_sq / ((x - c)**2 + w_sq)
    )
```

## Validation

- Parametrized test: for each model, generate random parameters (N=1000),
  verify vectorized output matches loop output to floating-point tolerance.
- Benchmark: time the old vs new implementation for N=1_000_000 parameter sets.

## Files to change

| File | Change |
|------|--------|
| `src/QDMpy/models.py` | Replace loop-based functions with vectorized versions |
| `tests/test_models.py` | Add parametrized correctness tests, optional benchmark |
## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.
