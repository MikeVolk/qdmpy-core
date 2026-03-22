# QEP-003: Unify Unit System

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | Nothing (QEP-001 superseded by QEP-011) |
| **Blocks** | QEP-005 |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |

## QEP-011 Impact

QEP-011 introduced a `freq_ghz` coordinate on the xarray DataArray, which is a step
toward GHz-everywhere. However, the core unit inconsistency problem remains:
`ODMRData.frequencies` still returns Hz, `1e9` conversions are still scattered across
`fit.py`, `data.py`, and `models.py`, and constants remain inconsistent.

## Motivation

Frequency values are inconsistently stored in GHz and Hz across the codebase.
At least 7 explicit GHz/Hz conversions are scattered across 3 files, and the
constants file itself is internally inconsistent:

```python
# constants.py - INCONSISTENT
AHYP_14N = 0.002158   # GHz
D_ZFS = 2.870e9        # Hz (!)
```

```python
# models.py - converts GHz constant to Hz
self.ahyp = AHYP_14N * 1e9

# fit.py - converts Hz data to GHz, then back to Hz for pygpufit
self.f_ghz = frequencies / 1e9
center_params *= 1e9      # GHz → Hz for pygpufit
center_results /= 1e9     # Hz → GHz back

# result.py - hardcoded values instead of constants
d_zfs = 2.87e9             # Hz, not using D_ZFS constant
zero_field_freq = 2.87     # GHz, hardcoded
```

Every conversion is a potential 1e9 error. The cognitive load of tracking which
unit each variable uses is unsustainable.


## Specification

### Decision: GHz Everywhere

All frequency values throughout the codebase will be in GHz. The only exception
is the pygpufit interface boundary, where conversion to Hz happens in a single,
clearly marked location.

### 1. Normalize constants.py

All frequency constants in GHz:

```python
"""Physical constants for NV diamond magnetometry.

Convention: All frequency values are in GHz. All magnetic field values are in T.
"""

GAMMA_NV = 28.024       # GHz/T — NV gyromagnetic ratio
D_ZFS = 2.870           # GHz — zero-field splitting
AHYP_14N = 0.002158     # GHz — 14N hyperfine coupling
AHYP_15N = 0.00303      # GHz — 15N hyperfine coupling
```

### 2. Remove Unit Conversion from Model.__init__

Models should store `ahyp` directly in GHz (no `* 1e9` conversion needed once
constants are in GHz):

```python
# Before
self.ahyp = AHYP_14N * 1e9  # Convert GHz to Hz

# After
self.ahyp = AHYP_14N  # Already in GHz
```

### 3. Centralize Hz Conversion in fit_frange() Only

Create a single conversion boundary in `FitManager.fit_frange()`:

```python
def fit_frange(self, ...):
    # --- GHz → Hz boundary (pygpufit requires Hz) ---
    frequencies_hz = self.frequencies * 1e9
    constraints_hz = self._convert_freq_constraints_to_hz(constraints)

    # ... call pygpufit with Hz values ...

    # --- Hz → GHz boundary ---
    results = self._convert_freq_results_to_ghz(raw_results)
```

All other code operates exclusively in GHz.

### 4. Fix result.py Hardcoded Values

Replace all hardcoded frequency values with constants:

```python
# Before
d_zfs = 2.87e9
zero_field_freq = 2.87

# After
from QDMpy.constants import D_ZFS
d_zfs = D_ZFS  # GHz
```

### 5. Document Convention

Add a module-level docstring to `constants.py` stating the unit convention.
Add inline comments to the pygpufit conversion boundary.

## Files Affected

- `src/QDMpy/constants.py` (normalize all constants to GHz)
- `src/QDMpy/models.py` (remove `* 1e9` conversions)
- `src/QDMpy/fit.py` (centralize Hz conversion at pygpufit boundary)
- `src/QDMpy/result.py` (use constants instead of hardcoded values)
- Tests that assert specific constant values

## Backwards Compatibility

This changes the numerical values of constants like `D_ZFS` (from `2.870e9` to
`2.870`). Any external code that imports these constants will see different values.

Since this is an internal overhaul and no external consumers depend on these
constants, this is acceptable. The API for `Measurement`, `FitResult`, etc.
remains unchanged — they already return GHz values.

## Verification

```bash
uv run pytest                    # All tests pass
uv run ruff check .              # No lint errors
# Manual check: grep for "1e9" should only appear in fit.py pygpufit boundary
grep -rn "1e9" src/QDMpy/ | grep -v "__pycache__"
```

## Rejection Alternatives

**Alternative: Hz everywhere.** Rejected because GHz is the natural unit for NV
diamond spectroscopy. The resonance frequency (~2.87 GHz) and hyperfine coupling
(~2.158 MHz = 0.002158 GHz) are more readable in GHz. Only pygpufit requires Hz.

**Alternative: Use a units library (pint/astropy.units).** Rejected as
over-engineering. The only unit boundary is GHz↔Hz at pygpufit. A library adds
dependency weight and runtime overhead for a problem solvable by convention.
