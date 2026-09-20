# QEP-018: Fix Width Unit Conversion Bug at GPU Boundary

**Status:** Implemented
**Priority:** Critical
**Affects:** `models.py`, `fit.py`, `guess.py`

## Problem

The `width` parameter is a frequency-domain quantity (Lorentzian linewidth in GHz)
but is **not** listed in `frequency_parameters` for any model. This causes a unit
mismatch at the pyGpufit boundary: `center` is converted from GHz to Hz, the
frequency axis (`user_info`) is converted to Hz, but `width` is left in GHz.

### The Lorentzian formula

All three models use the same structure:

```python
width_squared = p[1] * p[1]        # width²
aux = x - p[0]                      # x - center
dip = contrast * width_squared / (aux * aux + width_squared)
```

For this to produce physically correct results, `width`, `center`, and `x` must
share the same units. Currently at the GPU call site:

| Quantity | Unit | Source |
|----------|------|--------|
| `x` (user_info) | Hz | `freq * 1e9` (fit.py:691) |
| `center` | Hz | converted at fit.py:681-683 |
| `width` | **GHz** | **not converted** |

With `width` ~0.001 GHz and `x` ~2.87e9 Hz:
- `width² = 1e-6`
- `(x - center)² ≈ (1e6)² = 1e12` (near resonance)
- `dip ≈ contrast * 1e-6 / 1e12 ≈ 0` — the resonance dip vanishes

The same mismatch affects constraints: `width_min=0.0001, width_max=0.005` (GHz)
are passed to the GPU without conversion via `ConstraintManager.to_array()`.

### Affected code paths

1. **fit.py:681-683** — `fit_frange()` only converts parameters in
   `self._model.frequency_parameters`, which excludes `width`.
2. **fit.py:717-721** — `reshape_results()` only converts back parameters in
   `frequency_parameters`.
3. **fit.py:124-131** — `ConstraintManager.to_array()` only converts constraints
   for `frequency_parameters`.
4. **models.py** — All three model classes return `frequency_parameters = ['center']`,
   omitting `width`.

### Why it may appear to work

If the GPU kernel internally normalizes or uses a different parameterization than
the Python model functions, results could appear plausible. However, the Python
model functions (`esr14n`, `esr15n`, `esrsingle`) clearly use width in the same
units as x, confirming the bug.

## Proposed Fix

### Option A (minimal, recommended): Add `width` to `frequency_parameters`

```python
# In ESR14N, ESR15N, ESRSINGLE:
@property
def frequency_parameters(self) -> list[str]:
    return ['center', 'width']
```

This requires zero changes to the conversion logic in `fit.py` or
`ConstraintManager` — they already iterate over `frequency_parameters`.

### Option B: Convert everything to Hz at a single boundary

Replace the per-parameter conversion with a blanket conversion of all
frequency-unit parameters at the FitManager boundary. This is more robust
but requires a way to tag each parameter's unit — which the `units` property
on Model already provides.

```python
# In fit_frange():
for idx, param_name in enumerate(self.model_params_unique):
    if self._model.units[param_name] == 'GHz':
        initial_parameters_reshaped[:, idx] *= 1e9
```

## Validation

1. Unit test: create a known Lorentzian in Hz, fit it, assert recovered width
   matches to <1% relative error.
2. Integration test: compare B111 maps before/after fix using reference data
   from `~/github/QDMpy_old`.
3. Verify constraint bounds in Hz space are physically reasonable
   (100 kHz – 5 MHz linewidth).

## Migration

- This is a **bug fix**, not an API change.
- Existing saved results (NPZ files) will have widths in GHz, which is correct
  for the public API. Only the GPU boundary is affected.
- Users who have run fits with the current code should re-run them; fitted
  width values are likely meaningless.

## Files to change

| File | Change |
|------|--------|
| `src/QDMpy/models.py` | Add `'width'` to `frequency_parameters` in all 3 models |
| `tests/test_models.py` | Assert width is in frequency_parameters |
| `tests/test_fit.py` | Add round-trip unit test for width |
