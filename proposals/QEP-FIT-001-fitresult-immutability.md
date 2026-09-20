# QEP-FIT-001 — FitResult Immutability and Explicit Dimensions

**Status:** Draft
**Created:** 2026-02-22
**Severity:** CRITICAL (C-1) + HIGH (H-6)
**Module:** `fitting/result.py`, `odmr/data.py`, `odmr/folding.py`
**Revised:** 2026-09-19 -- drift update; scope extended to one immutability rule for FitResult, ODMRData and FoldedODMR; section 4 superseded

---

## Motivation

`FitResult` is the most-used object in the codebase — every downstream
analysis (B111, delta_resonance, parameter maps, persistence) depends on it.
Two issues create correctness risks:

1. **Not truly frozen.** `model_config` lacks `frozen=True`, so fields can be
   reassigned after construction (`result.parameters = {}`). The
   `flags.writeable = False` half-measure protects array *contents* but not
   the dict reference. Cache (`_b_field_cache`, `_delta_resonance_cache`,
   `_b111_cache`) has no invalidation — reassigning `parameters` silently
   returns stale cached results.

2. **Hardcoded `n_pol=2` in `_normalize_resonance_shape`.** When the center
   parameter is 2D (shape `(combined, n_pixel)`), the method assumes
   `n_pol=2` and divides `shape[0]` to get `n_frange`. A single-polarity
   dataset silently produces wrong B111 values. The root cause is that
   `FitResult` does not know its polarity/frange counts — it guesses from
   array shapes.

3. **Immutability is enforced three incompatible ways** (added 2026-09-19,
   from the 2026-08-30 bugs & tech-debt review):
   - `FitResult`: arrays are read-only, but fields can be reassigned (no
     `frozen=True`).
   - `ODMRData` (`odmr/data.py:46`) and `FoldedODMR` (`odmr/folding.py:174`):
     `frozen=True`, but the wrapped `xr.DataArray` / `NDArray` buffers stay
     writeable, so "frozen" means only that fields can't be reassigned.
   - `FoldingSettings`, `RefitSettings`, processors: frozen pydantic configs
     with no arrays, the one place where frozen means what it says.

### Status (2026-09-19)

Items 1 and 2 are still open: `result.py:57` has no `frozen=True`, and
`_normalize_resonance_shape` still guesses `n_pol = 2` (`result.py:300`).
Since 2026-08-30 `model_post_init` *copies* a writeable parameter array
before marking it read-only (it used to freeze the caller's array in place),
which makes the original section 4 below obsolete.

## GUI Integration Requirements

1. **Touchpoints.** `app/measurement_vm.py` reads `FitResult` parameter maps,
   `b111_*`, `chi2` and `metadata`, and `FoldedODMR.fold_residual` /
   `folded_spectrum`; `widgets/fold_diagnostics.py` reads the `FoldedODMR`
   arrays. All of these are reads.
2. **Persisted data.** `.qdm` files do not change format: `n_pol`/`n_frange`
   are derived from the saved array shapes on load.
3. **User-facing behaviour.** None expected. Any GUI code that writes into a
   result array (for example to mask pixels for display) now raises
   `ValueError: assignment destination is read-only`, and must copy first.
   Audit `measurement_vm.py` for in-place writes before implementing.
4. **Acceptance.** Load -> fit -> fold -> refit -> save `.qdm` -> reload ->
   all maps render identically, and GUI tests pass.

## Current Code

```python
# result.py:55 — no frozen=True
model_config = ConfigDict(arbitrary_types_allowed=True)

# result.py:284-288 — hardcoded n_pol=2
elif resonance.ndim == 2:
    n_pol = 2
    n_frange = resonance.shape[0] // n_pol
    n_pixels = resonance.shape[1]
    resonance = resonance.reshape((n_pol, n_frange, n_pixels))
```

## Proposed Changes

### 1. Add `frozen=True` to model_config

```python
model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
```

Any code that currently reassigns fields must use `model_copy(update={...})`
instead.

### 2. Add explicit `n_pol` and `n_frange` fields

```python
class FitResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    parameters: dict[str, NDArray]
    scan_dimensions: tuple[int, int]
    pixel_spacing: float = Field(gt=0)
    model_name: str
    n_pol: int = Field(gt=0)
    n_frange: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

These are always known at construction time in `FitManager.fit()`:

```python
# manager.py:235 — add n_pol and n_frange
return FitResult(
    parameters=parameters,
    scan_dimensions=scan_dimensions,
    pixel_spacing=pixel_spacing,
    model_name=model.name,
    n_pol=n_pol,
    n_frange=n_frange,
    metadata=metadata,
)
```

### 3. Simplify `_normalize_resonance_shape`

Remove all dimension guessing. Use `self.n_pol` and `self.n_frange` directly:

```python
def _normalize_resonance_shape(self, resonance: NDArray) -> tuple[NDArray, int]:
    """Reshape center array to (n_pol, n_frange, n_pixels) using known dimensions."""
    expected = self.n_pol * self.n_frange
    if resonance.ndim == 3 and resonance.shape[:2] == (self.n_pol, self.n_frange):
        return resonance, resonance.shape[2]
    n_pixels = resonance.size // expected
    return resonance.reshape(self.n_pol, self.n_frange, n_pixels), n_pixels
```

### 4. Remove `model_post_init` mutation (superseded 2026-09-19)

*Superseded:* the validator below freezes the **caller's** arrays in place,
which is the bug fixed on 2026-08-30. Keep the current copy-then-freeze in
`model_post_init`; with `frozen=True` it must assign through
`object.__setattr__`, as it already does. Original text:

The `flags.writeable = False` loop mutates parameter arrays in-place during
construction. With `frozen=True`, this becomes a Pydantic validation step
instead:

```python
@field_validator('parameters')
@classmethod
def freeze_arrays(cls, v: dict[str, NDArray]) -> dict[str, NDArray]:
    for arr in v.values():
        if isinstance(arr, np.ndarray):
            arr.flags.writeable = False
    return v
```

### 5. One immutability rule (decided 2026-09-19)

All three result containers follow one rule, stated in
`memory/architecture.md` and in each class docstring:

> Fields cannot be reassigned (`frozen=True`), and no array reachable from
> the object is writeable through it.

How each class meets it depends on who owns the arrays and how large they
are:

| Class | Arrays | How they are made read-only | Cost |
|---|---|---|---|
| `FitResult` | parameter maps, ~50-440 MB | copy if writeable, then read-only (current behaviour) | one copy per new result |
| `ODMRData` | raw stack, 1.9-3.8 GB at 1200x1920x51 | read-only **view** of the caller's buffer, no copy | none |
| `FoldedODMR` | folded/antisymmetric spectra, ~0.9-1.8 GB | read-only view; `SpectralFolder` owns the buffers it creates | none |

For `ODMRData`, a validator replaces `data` with
`data.copy(deep=False)` whose `.values` view has `flags.writeable = False`.
qdmpy cannot then mutate a stack through an `ODMRData`, but a caller that
kept its own reference to the original numpy array still can, and that is
documented. A deep copy was considered and rejected: 1.9-3.8 GB extra peak
memory at load and after every processor step, for protection the processors
do not need (they already return new arrays).

## Migration

- `FitManager.fit()` must pass `n_pol` and `n_frange` (trivial — already
  computed locally).
- `FitResult.load_results()` must reconstruct `n_pol`/`n_frange` from saved
  parameter shapes or save them explicitly in the NPZ.
- `testing.py` helpers (`make_synthetic_fit_result`) must pass the new fields.
- Tests that construct `FitResult` directly need the two new required fields.
- `FitResult` construction sites to update: `fitting/manager.py`
  (`_assemble_result`), `io/qdm.py:454` (`.qdm` load, where `n_pol`/`n_frange`
  come from the saved array shapes), `testing.py:316`, and `load_results`
  (`result.py:753`).
- Any code or test that writes into `odmr_data.data.values[...]` or a
  `FoldedODMR` array must copy first. Known case:
  `tests/odmr/test_processors.py` writes into `sample_odmr_data` in place.

## Alternatives Considered

- **Use `@dataclass(frozen=True)` instead of Pydantic** — Pydantic provides
  validation (scan_dimensions > 0, pixel_spacing > 0) that we'd lose.
- **Add cache invalidation** — Complex and error-prone. True immutability
  makes invalidation unnecessary.

## Test Plan

- [ ] Verify `FitResult` rejects field assignment post-construction
- [ ] Verify `_normalize_resonance_shape` uses `n_pol`/`n_frange` (no guessing)
- [ ] Test single-polarity data produces correct `delta_resonance`
- [ ] Verify `load_results` round-trips `n_pol`/`n_frange`
- [ ] Verify all existing tests still pass with new required fields
- [ ] Verify `ODMRData(...).data.values` and every `FoldedODMR` array are
      read-only, and that constructing `ODMRData` does not copy the buffer
      (`np.shares_memory` with the input)
- [ ] Verify `FitResult` does not freeze the caller's arrays (regression for
      2026-08-30)
