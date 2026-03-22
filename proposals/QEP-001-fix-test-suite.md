# QEP-001: Fix Test Suite and Establish Data Convention Ground Truth

| Field | Value |
|-------|-------|
| **Status** | Superseded by QEP-011 |
| **Priority** | P0 (BLOCKING) |
| **Complexity** | M |
| **Depends on** | Nothing |
| **Blocks** | ~~QEP-002, QEP-003, QEP-004, QEP-005, QEP-007, QEP-009, QEP-010~~ |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |
| **Superseded** | 2026-02-15 |

## Superseded

**This QEP has been fully superseded by QEP-011 (Adopt xarray for ODMR data
representation).**

QEP-011 structurally resolved the root causes addressed by this proposal:

1. **Axis ordering contradiction**: Eliminated entirely. xarray uses named dimensions
   `(polarity, freq_range, y, x, freq_idx)`, making axis confusion impossible.
2. **Model parameter naming/ordering**: Test fixtures were updated as part of QEP-011
   to match the actual model implementations.
3. **ConstraintManager API mismatch**: Tests were rewritten to use the correct Pydantic
   `ModelConstraintsSettings` objects.
4. **Miscellaneous failures**: All resolved. The test suite now has 256 passing tests
   with 0 failures.

All downstream QEPs that previously depended on QEP-001 now have no blocking
prerequisite (or depend directly on QEP-011 where applicable).

---

## Original Motivation (historical)

44 of 285 tests fail because tests were written against interfaces that don't match
the implementation. The failures fall into several root-cause categories:

1. **Axis ordering contradiction (~15 failures):** Code uses shape
   `(n_pol, n_frange, n_pixels, n_freq)` but docstrings claim
   `(n_pol, n_frange, n_freq, n_pixels)`. Tests follow the wrong docstrings.
2. **Model parameter naming/ordering (~12 failures):** Tests expect ESR14N params
   `["contrast", "center", "width_0", ...]` but the actual model defines
   `["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"]`.
3. **ConstraintManager API mismatch (~6 failures):** Tests pass `dict` but code
   expects a Pydantic `ModelConstraintsSettings` object.
4. **Miscellaneous (~11 failures):** FitManager param index assertions, numba edge
   cases, measurement mock misalignment, ODMR loader mocking issues.

No refactoring can be safely performed without a green test suite. This QEP is the
prerequisite for all other work.

## Specification

### Phase 1: Canonical Axis Convention

Establish and document the canonical data shape:

```
(n_pol, n_frange, n_pixels, n_freq)
```

- axis 0: polarities (typically 2)
- axis 1: frequency ranges (typically 2)
- axis 2: spatial pixels (flattened, typically ~4M)
- axis 3: frequency sweep points (typically ~50)

Fix incorrect docstrings in:
- `src/QDMpy/guess.py` - function docstrings claiming freq before pixels
- `src/QDMpy/odmr/io.py` - `BaseLoader` docstring
- `src/QDMpy/odmr/odmr.py` - `ODMR.load_data` docstring

### Phase 2: Fix guess.py Test Fixtures

The `sample_odmr_data` fixture creates `(2, 3, 100, 10)` intending 100=freq,
10=pixels. Since code treats axis 2 as pixels:

- Change fixture shape to `(2, 3, 10, 100)` (10 pixels, 100 freq points)
- Update all expected output shapes to `(2, 3, 10)` (one result per pixel)
- Verify `get_peak_list` and `get_model_by_peaks` produce correct outputs

### Phase 3: Fix Model Parameter Tests

Update `test_models.py` assertions to match actual ESR14N implementation:

- Parameter names: `["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"]`
- Parameter count: 6 (not the assumed count in tests)
- `ahyp` value: `AHYP_14N * 1e9` (stored in Hz internally)
- Fix `n_peaks` and `param_count` assertions for each model variant

### Phase 4: Fix ConstraintManager Tests

In `test_fit.py::TestConstraintManager`:

- Replace `dict` constructor arguments with `ModelConstraintsSettings()` Pydantic
  objects
- Ensure constraint bounds, types, and defaults match the Pydantic schema
- Verify `get_constraint_array()` output shape and values

### Phase 5: Fix FitManager Tests

- Fix `_param_idx` assertions to match correct model parameter ordering
- Fix error message string expectations
- Align mock setups with actual `FitManager` interface
- Verify `fit_frange()` integration with corrected constraints

### Phase 6: Fix Remaining Failures

- numba edge cases: handle empty/zero arrays in `@njit` functions
- `test_measurement.py`: align mock data shapes with canonical convention
- `tests/odmr/test_data.py`, `tests/odmr/test_io.py`: fix loader mock returns

## Files Affected

### Source (docstrings only)
- `src/QDMpy/guess.py`
- `src/QDMpy/odmr/io.py`
- `src/QDMpy/odmr/odmr.py`

### Tests
- `tests/test_guess.py`
- `tests/test_models.py`
- `tests/test_fit.py`
- `tests/test_measurement.py`
- `tests/odmr/test_data.py`
- `tests/odmr/test_io.py`

## Backwards Compatibility

No API changes. Only docstring corrections in source and test fixture corrections.
All changes make tests match the existing implementation, not the other way around.

## Verification

```bash
uv run pytest                              # 0 failures (currently 44)
uv run pytest --tb=short 2>&1 | tail -5    # Confirm clean run
uv run ruff check .                        # No new lint errors
```

## Rejection Alternatives

**Alternative: Change code to match docstrings.** Rejected because the code's
convention `(n_pol, n_frange, n_pixels, n_freq)` is more natural for the dominant
operation (fitting per-pixel across frequencies) and matches the numba kernel
signatures already in use.
