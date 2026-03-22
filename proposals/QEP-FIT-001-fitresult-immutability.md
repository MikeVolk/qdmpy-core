# QEP-FIT-001 — FitResult Immutability and Explicit Dimensions

**Status:** Draft
**Created:** 2026-02-22
**Severity:** CRITICAL (C-1) + HIGH (H-6)
**Module:** `fitting/result.py`

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

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

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

### 4. Remove `model_post_init` mutation

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

## Migration

- `FitManager.fit()` must pass `n_pol` and `n_frange` (trivial — already
  computed locally).
- `FitResult.load_results()` must reconstruct `n_pol`/`n_frange` from saved
  parameter shapes or save them explicitly in the NPZ.
- `testing.py` helpers (`make_synthetic_fit_result`) must pass the new fields.
- Tests that construct `FitResult` directly need the two new required fields.

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
