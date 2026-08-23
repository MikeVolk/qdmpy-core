# QEP-071 - Refit Pipeline Consistency (freq_cutoff / constraint overrides in fit_frange)

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P1 |
| **Complexity** | S |
| **Depends on** | QEP-070 (fit pipeline unification) |
| **Reverses (scope only)** | QEP-070's explicit "refit.py's use of fit_frange ... keeps working untouched" goal |
| **Author** | QDMpy Team |
| **Created** | 2026-08-23 |

---

## Motivation

A math/architecture review (`docs/reviews/2026-08-22-math-and-architecture-review.md`,
finding F3, HIGH) found that `qdmpy.fitting.refit`'s outlier-refit path silently bypasses
two things the original fit applies per frequency range:

1. **`freq_cutoff` trimming.** `FitManager.fit_frange()` — the method `refit.py`'s
   `_refit_pass()` calls to refit outlier pixels — always fit against the full, uncut
   frequency axis, regardless of any `freq_cutoff` configured on the manager. Confirmed at
   `measurement_workflows.py`'s `refit_measurement_result`: the `FitManager` used for refit
   is constructed *with* `freq_cutoff` forwarded, but `fit_frange()` never applied it.
2. **Constraint overrides.** The per-range mT-center-window (`_effective_constraints_for_range`)
   and the folded-fit contrast/offset overrides (`FOLDED_CONSTRAINT_OVERRIDES`) were computed
   only inside `_fit_prepared()` (used by `fit()`/`fit_folded()`); `fit_frange()` used the
   manager's plain, unmodified `self.constraints`.

Net effect: whenever `freq_cutoff` is configured, or a folded fit later needs outlier
pixels refit, those pixels are silently fit against a different objective (different
frequency range and/or different constraint bounds) than every other pixel in the same
map. No error is raised; the result is just quietly inconsistent.

This was not an oversight in QEP-070 — that proposal explicitly scoped it out: "Zero public
API change: `fit()`, `fit_folded()`, `fit_frange()` signatures ... unchanged. `refit.py`'s
use of `fit_frange` (legacy 5-element list return) keeps working untouched," and listed
"Refit contract drift" as a risk to *avoid* (`fit_frange` becomes a thin adapter, not a
re-derivation). `tests/test_refit.py` encoded that as a frozen invariant: existing tests
only assert `freq_cutoff` is *forwarded as a kwarg* to the refit machinery, never that it's
*applied*.

QDMpy is unreleased, so `fit_frange()`'s signature is not a compatibility constraint this
QEP needs to preserve — deliberately reversing QEP-070's scope boundary here, rather than
adding a parallel method, is the simpler and more honest fix.

## Goals

- `fit_frange()` applies the same per-range preprocessing `_fit_all_franges()` already uses:
  `freq_cutoff` trimming and constraint-override layering (mT center window, plus any
  caller-supplied `constraint_overrides`).
- `refit.py`'s `_refit_pass()` passes the range coordinate (`irange`/`n_frange`) and, when
  refitting a folded fit's outliers (`fit_result.metadata["folded_fit"]`), the same
  `FOLDED_CONSTRAINT_OVERRIDES` that `fit_folded()` applied originally.
- Close the test-coverage gap: add real (non-mocked) regression tests proving `freq_cutoff`
  and folded overrides are actually applied during refit, not just forwarded as kwargs.

## Non-goals

- No further unification of `fit_frange()` into `_fit_prepared()` — refit fits an arbitrary
  pixel *subset* with externally-supplied initial guesses (neighbor-derived, not
  re-guessed), which doesn't fit `_fit_prepared()`'s whole-array/auto-guessing shape.
  `_fit_all_franges()` is untouched; it was already correct.
- No change to refit's outlier-detection or neighbor-guess logic (`identify_outlier_pixels`,
  `compute_neighbor_guesses`) — those are unaffected by this fix.
- No physics/algorithm changes — this only makes refit consistent with whatever the
  original fit was configured to do.

## GUI Integration Requirements

1. **Core API/data contracts**: `fit_frange()`'s signature changed (added required
   `irange`/`n_frange`, optional `constraint_overrides`), but this is an internal
   `FitManager` method. Confirmed via grep that `qdmpy-gui` only depends on
   `RefitSettings` and `Measurement.refit_outliers`, neither of which changed — no GUI code
   references `FitManager.fit_frange` directly.
2. **GUI settings/controls**: none required.
3. **State/metadata for rendering**: unchanged — `FitResult.metadata["refit_info"]` schema
   is the same; refit results are now just numerically consistent with the original fit.
4. **Error/progress behavior**: unchanged exception types. `fit_frange()` now also raises
   `DataValidationError` if `freq_cutoff` is incompatible with `n_frange` (matching
   `_fit_prepared`'s existing validation) — in practice unreachable via refit, since the
   same manager's `freq_cutoff` was already validated during the original `fit()` call.
5. **Acceptance / no-GUI-impact rationale**: internal-only change behind an unchanged
   `Measurement`-level surface. No GUI code changes expected or required.

---

## Design

`FitManager.fit_frange()` (`src/qdmpy/fitting/manager.py`) is extended in place — no
parallel method:

```python
def fit_frange(
    self,
    data: NDArray,
    freq: NDArray,
    initial_parameters: NDArray,
    *,
    irange: int,
    n_frange: int,
    constraint_overrides: Mapping[str, ConstraintOverride] | None = None,
) -> list[Any]:
```

`irange`/`n_frange` are required keyword-only args — no default, since a silent
`irange=0, n_frange=1` default would be wrong for any multi-frange caller and
`freq_cutoff` interacts with exactly this coordinate. Body mirrors
`_fit_all_franges()`'s per-range preprocessing: validate `freq_cutoff` against
`n_frange`, compute `base = self._base_constraints_with_overrides(model, constraint_overrides)`,
then `effective = self._effective_constraints_for_range(base, freq)` using the **full**,
pre-cutoff `freq` (the mT-window branch decision needs the uncropped min/max — same
ordering `_fit_all_franges` uses), then crop via `self._apply_freq_cutoff_for_range(data,
freq, irange, n_frange)`, then delegate to `self._run_backend_fit(...)`. Return shape is
unchanged (5-element list).

`_FOLDED_CONSTRAINT_OVERRIDES` is renamed to `FOLDED_CONSTRAINT_OVERRIDES` (dropping the
leading underscore) since it's now used from `qdmpy.fitting.refit`, not just internally.

`refit.py`'s `_refit_pass()` computes `constraint_overrides = FOLDED_CONSTRAINT_OVERRIDES
if fit_result.metadata.get("folded_fit") else None` once per pass (the current
`FitResult`'s `"folded_fit"` metadata key, set by `fit_folded()` and propagated through
refit passes), and passes `irange=irange, n_frange=n_frange, constraint_overrides=...`
into each `fit_frange()` call.

---

## Implementation Plan

Single phase (small, cohesive change):

1. Extend `FitManager.fit_frange()`'s signature and body as above; rename
   `FOLDED_CONSTRAINT_OVERRIDES`.
2. Update `refit.py`'s `_refit_pass()` call site.
3. Update the three other `fit_frange()` call sites for the new required kwargs:
   `tests/integration/test_gpufit_consistency.py`, `tests/test_fit.py::test_fit_frange_mocked`.
4. Update `tests/test_refit.py`'s mocked `fit_frange` helpers (`_make_mock_fm`,
   `counting_fit_frange`) for the new signature.
5. Promote `RecordingFitBackend` (previously a private `test_fit.py`-only fixture) to
   `src/qdmpy/testing.py` alongside `FakeFitBackend`, extended to also record
   `constraints`/`constraint_types`.
6. Add two new regression tests to `tests/test_refit.py`: `test_refit_applies_freq_cutoff`
   and `test_refit_applies_folded_constraint_overrides`, both using a real `FitManager` +
   `RecordingFitBackend` (no mocks) to assert the backend actually received the cropped
   frequency axis / overridden constraint bounds.

## Test Plan

- `tests/test_refit.py::test_refit_applies_freq_cutoff` — real `FitManager` constructed
  with `freq_cutoff={"low": {"max": ...}}`, `RecordingFitBackend`; asserts the recorded
  `freq_ghz` for the refit call is cropped to the configured bound and shorter than the
  uncut axis.
- `tests/test_refit.py::test_refit_applies_folded_constraint_overrides` — same setup with
  `fit_result.metadata = {"folded_fit": True}`; asserts the recorded constraint bounds for
  `offset` match `FOLDED_CONSTRAINT_OVERRIDES["offset"]`, not the manager's plain base
  bounds.
- Both new tests manually verified to **fail** against the pre-fix `fit_frange()` body
  (temporarily reverted to call `self._run_backend_fit(data, freq, initial_parameters,
  model, self.constraints)` directly) and **pass** after — confirms they exercise the bug,
  not just the plumbing.
- Full regression: `uv run pytest` (1006 passed, 20 skipped [torch, no `gpu` extra
  installed], 86.70% coverage), `uv run ruff check .` (168 pre-existing hits, none new),
  `uv run ruff format --check .`, `uv run ty check src/qdmpy` (clean).

## Risks and Mitigations

1. **Other undiscovered `fit_frange()` callers outside the test suite/refit.py.**
   Mitigation: grepped the whole branch — only `refit.py` calls it in production code; the
   two test call sites are updated alongside.
2. **`freq_cutoff` validation now runs on every refit pass, not just the original fit.**
   Mitigation: harmless — the same manager's `freq_cutoff` was already validated against
   the same `n_frange` during the original `fit()`/`fit_folded()` call; this is a cheap,
   idempotent re-check, not new failure surface in practice.

## Alternatives Considered

1. **Add a parallel method (e.g. `fit_range_for_refit`) and keep `fit_frange()` frozen.**
   Rejected: the package is unreleased, so preserving `fit_frange()`'s exact legacy contract
   has no real payoff — it would mean two near-duplicate methods with no call site ever
   using the old one except tests, and a needless `_FOLDED_CONSTRAINT_OVERRIDES` cross-module
   coupling story to justify the split.
2. **Route `refit.py` through `_fit_prepared()` directly (finish A1 fully).** Rejected:
   `_fit_prepared()` guesses initial parameters from data; refit already has
   neighbor-derived guesses and fits an arbitrary pixel subset, not the whole array — the
   shapes don't match without a much larger refactor for no additional benefit over
   extending `fit_frange()` directly.
