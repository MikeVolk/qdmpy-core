# QEP-FIT-004 — B111 Physics Deduplication

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-7)
**Module:** `fitting/result.py`, `odmr/analysis.py`

---

## Motivation

The B111 calculation (splitting → magnetic field via `GAMMA_NV`) exists in two
places:

1. `FitResult._compute_b111()` in `fitting/result.py:368-408`
2. `b111_from_dip_positions()` (or equivalent) in `odmr/analysis.py`

Both implement the same physics:

```
dB = (f_high - f_low) / 2 / GAMMA_NV * 1e6   # µT
remanent = (neg + pos) / 2
induced  = (neg - pos) / 2
```

If one is updated (e.g. a sign convention fix) and the other is not, the two
code paths silently diverge. This has already happened once during the QEP-025
coordinate relabelling.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Proposed Changes

### 1. Extract canonical B111 functions into `fitting/physics.py`

```python
"""Physics formulae for NV-center magnetic field extraction.

All functions are pure (no side effects, no logging, no xarray).
Input/output in SI-adjacent units: frequencies in GHz, fields in µT.
"""
from numpy.typing import NDArray
from QDMpy.constants import GAMMA_NV


def delta_resonance(f_high: NDArray, f_low: NDArray) -> NDArray:
    """Signed frequency splitting → µT.

    Args:
        f_high: High-branch center frequencies (GHz), shape (...).
        f_low: Low-branch center frequencies (GHz), shape (...).

    Returns:
        Magnetic field contribution in µT, same shape as inputs.
    """
    return (f_high - f_low) / 2 / GAMMA_NV * 1e6


def b111_remanent(neg_diff: NDArray, pos_diff: NDArray) -> NDArray:
    """Remanent (permanent magnetisation) component in µT."""
    return (neg_diff + pos_diff) / 2


def b111_induced(neg_diff: NDArray, pos_diff: NDArray) -> NDArray:
    """Induced (bias-tracking) component in µT."""
    return (neg_diff - pos_diff) / 2
```

### 2. FitResult delegates to physics module

```python
# fitting/result.py
from QDMpy.fitting.physics import delta_resonance, b111_remanent, b111_induced

def _compute_delta_resonance(self) -> xr.DataArray:
    ...
    delta = delta_resonance(resonance[:, 1], resonance[:, 0])
    delta = delta.reshape(n_pol, height, width) * d  # sign per polarity
    ...

def _compute_b111(self) -> xr.Dataset:
    ...
    rem = b111_remanent(neg_diff, pos_diff)
    ind = b111_induced(neg_diff, pos_diff)
    ...
```

### 3. Remove duplicate in `odmr/analysis.py`

Replace with import from `fitting.physics`. If `odmr/analysis.py` has
additional analysis logic beyond B111, keep the file but delete the duplicated
formula.

## Migration

- No public API change — `FitResult.b111`, `.delta_resonance` unchanged.
- Any code importing B111 functions from `odmr/analysis.py` gets a one-line
  re-export or direct import change.
- `fitting/physics.py` is pure numpy — trivially testable with Hypothesis.

## Test Plan

- [ ] Property-based tests for `delta_resonance`: symmetric inputs → 0
- [ ] Property-based tests for `b111_remanent`/`b111_induced`: sum = neg_diff
- [ ] Verify `FitResult.b111` returns identical values after refactor
- [ ] Verify no remaining duplicate formula in codebase (`grep GAMMA_NV`)
