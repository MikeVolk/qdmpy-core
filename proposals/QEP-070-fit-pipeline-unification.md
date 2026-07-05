# QEP-070 - Unified Fit Pipeline (FitManager Decomposition + Folded-Path Unification)

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Priority** | P1 |
| **Complexity** | L |
| **Depends on** | QEP-068 (fit backend seam), QEP-069 (torch backend) |
| **Supersedes** | QEP-FIT-003 (fitmanager-decomposition), QEP-060 (internal-fit-path-unification) |
| **Blocks** | QEP-057 (kwarg tuple collapse becomes easier once `freq_cutoff` is a value object) |
| **Author** | QDMpy Team |
| **Created** | 2026-07-05 |

---

## Motivation

The 2026-07-03 architecture review (`.mike/architecture-review-findings.md`, finding #2)
ranked "one fit pipeline inside FitManager" as the top remaining candidate now that QEP-068
gave `FitManager` an injectable `FitBackend`. Five concrete problems, all in
`src/qdmpy/fitting/manager.py`:

1. **`fit()` (lines 319-430) is a god method.** Backend guard, input validation, data
   flattening, auto-model resolution, buffer allocation, a per-frange loop that itself does
   five things, transpose/reshape, quality metrics, and `FitResult` construction — nine jobs
   in one method with no independently testable seam between them.

2. **`fit_folded()` (738-850) re-implements the pipeline by hand and constructs a second
   `FitManager` inside itself** (832-838). It hand-derives the folded→fit-input conversion
   that `FoldedODMR.to_fit_inputs()` (`odmr/folding.py:185-206`) already provides — the
   refit path already uses `to_fit_inputs()` (`measurement_workflows.py:289`), so today the
   two call sites diverge on the exact same conversion. `fit_folded` then builds a
   throwaway `FitManager` purely to reuse `fit()`'s logic, meaning any future fix to `fit()`
   must be remembered twice.

3. **`_apply_mt_center_window_for_range()` (432-475) mutates the shared `ConstraintManager`
   mid-fit** via `set_constraint("center", ...)`, called once per frange inside `fit()`'s
   loop. This contradicts the class docstring's "stateless between calls" claim (54-57):
   after a fit with mT-mode `LOWER_UPPER` center constraints, `mgr.constraints["center"]`
   silently retains whichever frange fit last, and a second `fit()` call on the same
   manager starts from that leaked state instead of the original bounds.

4. **~110 lines of `freq_cutoff` dict parsing** (five private methods, manager.py:190-304)
   validate and apply an ad hoc `{'low'|'high': {'min': float|None, 'max': float|None}}`
   schema inline. It belongs in a frozen, independently testable value object, following the
   pattern already used for `FitBackendOptions` (`fitting/backends.py:50-58`).

5. **`_param_idx` carries undocumented aliases** (`"resonance"`→`"center"`,
   `"mean_contrast"`→`"contrast"`) with no deprecation path — a hidden API surface.

This QEP supersedes the two existing drafts that partially covered this ground
(QEP-FIT-003 for the decomposition, QEP-060 for folded/non-folded unification) because the
two problems are intertwined — you cannot cleanly extract a shared `_fit_prepared` path
without first decomposing `fit()`, and you cannot remove `fit_folded`'s second-manager
construction without a non-mutating constraint mechanism to carry its per-call overrides.
It also folds in the `ConstraintManager` mutation bug and the `FreqCutoff` extraction that
neither draft scoped.

## Goals

- Decompose `fit()` into small, independently unit-testable stages.
- `fit_folded()` becomes preparation (folded→absolute-GHz conversion, constraint overrides)
  plus delegation to the same internal execution path `fit()` uses — no second `FitManager`.
- Fix the mid-fit `ConstraintManager` mutation: per-range center-window bounds are computed
  and applied without mutating shared state, so the manager is actually stateless between
  calls (model resolution excepted, which is documented as one-time).
- Extract `freq_cutoff` parsing into a frozen `FreqCutoff` value object with unit test
  coverage independent of `FitManager`.
- Deprecate (not silently keep) the `_param_idx` aliases.
- Zero public API change: `fit()`, `fit_folded()`, `fit_frange()` signatures and the
  `FitResult` contract are unchanged. `refit.py`'s use of `fit_frange` (legacy 5-element
  list return) keeps working untouched.

## Non-goals

- No change to folded spectrum construction (QEP-011 scope) or B111/physics equations.
- No removal of `Measurement.fit_odmr` / `fit_folded_odmr` — the public/product-level
  distinction between folded and non-folded workflows stays (QEP-060's original position).
- No `.qdm`/serialization format changes.
- No collapse of the `(constraints, freq_cutoff, settings, gpu_available)` kwarg tuple
  threaded through `measurement_workflows.py` — that is QEP-057's scope; this QEP only
  makes it easier by turning `freq_cutoff` into a value object.
- No workflow-level forwarding of raw unfolded data for folded auto-model detection
  (would change today's behavior where `fit_folded_measurement_odmr` never passes
  `raw_data`); left as a follow-up.

## GUI Integration Requirements

1. **Core API/data contracts**: unchanged. `Measurement.fit_odmr`, `fit_folded_odmr`,
   `FitManager.fit`/`fit_folded`/`fit_frange` keep identical signatures and return types.
   `FitResult.parameters`/`.metadata`/`.scan_dimensions` schema is unchanged.
2. **GUI settings/controls**: none required. No new settings keys, no renamed defaults.
3. **State/metadata for rendering**: unchanged; `metadata["quality_metrics"]` and
   `metadata["folded_fit"]` keep the same keys and semantics.
4. **Error/progress behavior**: unchanged exception types and, with one documented
   exception (see Risks — error-ordering), unchanged messages. No new user-facing errors.
5. **Acceptance / no-GUI-impact rationale**: this is an internal refactor behind an
   unchanged public surface. Acceptance check: run `qdmpy-gui`'s fit action (folded and
   non-folded) against a real or fake dataset before/after this change and confirm
   identical parameter maps and identical error dialogs for the same bad inputs. No GUI
   code changes expected; parity tests in this QEP are the automated proxy for that check.

---

## Design

### 1. Shared internal execution path

```python
@dataclass(frozen=True)
class _PreparedFitInputs:
    flat_data: NDArray       # (n_pol, n_frange, n_pixel, n_freq)
    freq_ghz: NDArray        # (n_frange, n_freq)
    scan_dimensions: tuple[int, int]

@dataclass(frozen=True)
class _RangeFitOutputs:
    params: NDArray          # (n_frange, n_pol, n_pixel, n_params)
    states: NDArray
    chi2: NDArray
    iterations: NDArray
    exec_times: tuple[float, ...]

def _fit_prepared(
    self,
    prepared: _PreparedFitInputs,
    *,
    pixel_spacing: float,
    detection_data: NDArray | None = None,
    constraint_overrides: Mapping[str, ConstraintOverride] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> FitResult:
    ...
```

`fit()` becomes: guard → validate → `_prepare_data` → `_fit_prepared`. `fit_folded()`
becomes: guard → `folded.to_fit_inputs()` → validate → `_prepare_data` → `_fit_prepared(...,
constraint_overrides=_FOLDED_CONSTRAINT_OVERRIDES, extra_metadata={"folded_fit": True})`.

### 2. Non-mutating per-range constraints

`_mt_center_window_for_range(freq_ghz) -> tuple[float, float] | None` computes the window
purely (same branch logic as today). `_effective_constraints_for_range(base, freq_ghz)`
returns a new `dict[str, Constraint]` with `Constraint.with_updates(...)` applied — the
shared `ConstraintManager` is never mutated mid-fit.

### 3. Folded constraint overrides without a second manager

`fit_folded`'s current per-call contrast/offset overrides become a small frozen
`ConstraintOverride(vmin, vmax, constraint_type)` keyed by parameter *type*, applied via
`_base_constraints_with_overrides(model, overrides)` in the same non-mutating style —
replacing the "build a dict, construct a whole new FitManager" pattern.

### 4. `FreqCutoff` value object

`src/qdmpy/fitting/freq_cutoff.py`: frozen pydantic `FreqCutoffBounds`/`FreqCutoff` with
`from_raw()` (boundary normalizer preserving today's exact error messages),
`validate_for_n_ranges()`, `bounds_for_range()`, `apply_to_range()` — verbatim ports of the
five private methods being deleted from `manager.py`.

### 5. `to_fit_inputs()` gains coords

`FoldedODMR.to_fit_inputs()` gains the `polarity`/`freq_range` coords `fit_folded` attaches
today (additive; the refit path, which only reads `.values`/`.sizes`, is unaffected), so
both consumers of the folded→fit-input conversion use the exact same method.

---

## Implementation Plan

### Phase 0 - Characterization baseline (tests only)
Pin current behavior: `fit()`/`fit_folded()` output shapes and quality metrics against a
`FakeFitBackend`, backend-received-input identity across repeat fits, and per-frange mT
window values. These tests must keep passing byte-identically through every later phase.

### Phase 1 - `FreqCutoff` value object
Extract and delete the five private parsing methods from `manager.py`; `FitManager`
delegates to `FreqCutoff.from_raw()` at construction. No behavior change; public
constructor signature/docstring unchanged.

### Phase 2 - Non-mutating constraint machinery
Add pure projections in `constraints.py`; replace `_apply_mt_center_window_for_range`
(mutating) with `_mt_center_window_for_range` + `_effective_constraints_for_range` (pure).
`fit_frange()` becomes a thin adapter over a new `_run_backend_fit` helper. Intended
behavioral fix: `mgr.constraints` no longer leaks the last frange's window after `fit()`.

### Phase 3 - Decompose `fit()`; introduce `_fit_prepared`
Extract `_prepare_data`, `_resolve_model`, `_guess_parameters`, `_fit_all_franges`,
`_assemble_result`, and the shared `_fit_prepared` entry point. `fit()` shrinks to ~8 lines.
Per-stage unit tests added.

### Phase 4 - `fit_folded` becomes preparation + delegation
Route through `FoldedODMR.to_fit_inputs()` + `_fit_prepared(..., constraint_overrides=...,
extra_metadata={"folded_fit": True})`. Delete the hand-built conversion, the constraint
dict rebuild, and the second `FitManager` construction. Folded/non-folded parity tests
added (same effective backend inputs modulo the documented overrides).

### Phase 5 - Alias deprecation, docs, guardrails
`_param_idx` aliases raise `DeprecationWarning` before remapping (removal deferred one
release). Fix the class docstring's stateless-between-calls claim. Add a
sequential-statelessness regression test. Promote this QEP to Implemented; mark QEP-FIT-003
and QEP-060 Superseded.

---

## Test Plan

- Unit tests per extracted stage (`_prepare_data`, `_resolve_model`, `_guess_parameters`,
  `_fit_all_franges`, `_assemble_result`), all against `FakeFitBackend` — no mocks.
- Characterization/parity suite (`tests/test_fit_pipeline_parity.py`): `fit()` and
  `fit_folded()` byte-identical outputs before/after refactor; folded vs. non-folded
  produce identical backend-received arrays modulo the documented contrast/offset
  overrides; no second `FitManager` is constructed during `fit_folded`.
- Constraint-immutability regression: snapshot `mgr.constraints` before/after `fit()` under
  mT `LOWER_UPPER` settings — must be unchanged, while per-call backend arrays still show
  the correct asymmetric per-branch windows.
- `FreqCutoff` unit tests: message parity with today's `DataValidationError` text, numpy
  scalar coercion, empty→`None` normalization, min-points enforcement.
- Full regression: `tests/test_fit.py`, `tests/test_folded_fit.py`, `tests/test_refit.py`
  (refit's use of `fit_frange` — legacy 5-element return, manager-stored constraints,
  no mT window — must be untouched) run green at every phase boundary via
  `uv run pytest`.
- GUI smoke check per GUI Integration Requirements above.

## Risks and Mitigations

1. **Folded coords on `to_fit_inputs`'s `DataArray`** — refit path now receives
   `polarity`/`freq_range` coords it didn't have before. Mitigation: confirmed the refit
   path only consumes `.values`/`.sizes`; verified during Phase 4 implementation.
2. **`freq_cutoff` edge-case/message parity** — several existing tests assert on exact
   error-message fragments. Mitigation: `FreqCutoff.from_raw()` ports validation manually
   (not via pydantic field validators) to preserve exact wording; covered by dedicated
   `FreqCutoff` unit tests before `manager.py` is touched.
3. **Refit contract drift** — `fit_frange()` must keep returning the legacy 5-element list
   and using manager-stored (windowless) constraints. Mitigation: `test_refit.py` run at
   every phase boundary; `fit_frange` becomes a thin adapter, not a re-derivation.
4. **Error-ordering delta (documented, tolerated)**: for folded fits with an invalid
   `freq_cutoff` *and* auto-model resolution, the cutoff n-range error is now raised before
   model resolution (previously after) — same exception type and message, only earlier.
   Backend `.supports()` is likewise hoisted to once per fit instead of once per frange —
   same exception/message, earlier surfacing only.
5. **`_reshape_frange_results` squeeze semantics** for `n_pol == 1` are subtle; ported
   verbatim rather than "cleaned up" during the move.

## Alternatives Considered

1. **Remove `fit_folded()` entirely, route everything through `fit()` with a folded flag.**
   Rejected: worse readability at the `Measurement` layer, larger public-surface churn,
   and QEP-060's original design explicitly kept the folded/non-folded distinction at the
   product level while unifying only the internal execution path.
2. **Keep the two drafts (QEP-FIT-003, QEP-060) separate and implement them independently.**
   Rejected: the decomposition and the folded-path unification depend on each other (see
   Motivation); implementing them separately would mean touching `fit()`'s internals twice
   and re-deriving the same design questions about constraint-passing across both efforts.
3. **Fix the `ConstraintManager` mutation bug as a standalone patch, deferring the rest.**
   Rejected: the mutation fix is small in isolation, but the non-mutating projection it
   requires (`_effective_constraints_for_range`) is exactly the mechanism `fit_folded` needs
   to drop its second-`FitManager` construction — doing them together avoids building the
   same abstraction twice.

## Follow-ups (out of scope here)

- Forward raw unfolded data from `fit_folded_measurement_odmr` for auto-model detection
  (a behavior change, needs its own decision).
- Collapse the `(constraints, freq_cutoff, settings, gpu_available)` kwarg tuple across the
  seven workflow signatures into a `FitOptions` value object (QEP-057).
