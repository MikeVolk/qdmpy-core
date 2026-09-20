# QEP-074 - FitManager Module Split

| Field   | Value |
|---------|-------|
| Status  | Draft |
| Created | 2026-09-19 |
| Scope   | `qdmpy.fitting` (`manager.py`, `constraints.py`, new `fit_inputs.py`, `range_fit.py`, `result_assembly.py`) |
| Depends | QEP-070 (fit pipeline unification, Implemented) |
| Related | QEP-FIT-001 (result.py), QEP-FIT-005 (models.py) |

---

## Summary

`fitting/manager.py` is 987 lines, over the 800-line hard limit in
`CLAUDE.md` and well over the 600-line module guideline in
`.claude/rules/common/architecture.md`. This QEP moves three cohesive groups
out of `FitManager` into collaborators and functions, and replaces the five
constraint delegator methods with a single `constraints` property. The fit
pipeline itself is not redesigned -- QEP-070 already did that.

## Motivation / Problem statement

`FitManager` is the most important class in the package and the hardest to
read. After QEP-070 the *pipeline* is clean, but everything still lives in one
file: input validation, per-range backend execution, result assembly, the
constraint API, the folded path, and eight properties.

Three concrete costs:

1. **Reviewability.** A change to per-range constraint handling touches the
   same file as `.qdm` metadata assembly. `git log` on the file is unreadable
   as a history of either.
2. **Testability.** `_fit_all_franges`, `_run_backend_fit` and
   `_assemble_result` are private methods reached only through `fit()` or a
   constructed `FitManager`; tests hold a full manager to exercise array
   reshaping.
3. **The constraint API is five near-identical delegators.** Each repeats the
   same "model not yet resolved" guard and forwards to
   `ConstraintManager`, and `set_constraints` additionally fans `contrast`
   out to `contrast_0..n` -- logic that belongs with the constraints, not
   with the fit orchestrator.

### Current state

| Group | Members | Approx. lines |
|---|---|---|
| Input preparation | `_PreparedFitInputs`, `_validate_inputs`, `_prepare_data`, `_validate_freq_cutoff_for_n_ranges`, `_apply_freq_cutoff_for_range` | ~110 |
| Per-range execution | `_RangeFitOutputs`, `_fit_all_franges`, `_fit_prepared`, `_run_backend_fit`, `_effective_constraints_for_range`, `_mt_center_window_for_range`, `_base_constraints_with_overrides`, `fit_frange`, `_reshape_frange_results` | ~330 |
| Result assembly | `_assemble_result` | ~80 |
| Constraint API | `set_constraints`, `set_free_constraints`, `constraints`, `get_constraints_array`, `get_constraint_types`, `_param_idx` | ~125 |
| Orchestration | `__init__`, `fit`, `fit_folded`, `_resolve_model`, `_resolve_auto_model`, `_guess_parameters`, properties, `__repr__` | remainder |

External callers of the parts being moved: `fitting/refit.py:378`
(`fit_frange`), `measurement_workflows.py:416` (`fit_folded`), and the
`constraints=` dicts that `Measurement.fit_odmr` forwards. qdmpy-gui reaches
all of this through `Measurement`, except `fitting.constraints._mt_to_absolute_ghz`,
which it imports directly (`app/fit_constraints_vm.py`) and which does not move.

## Goals

- `fitting/manager.py` under 600 lines; no new module over 400.
- Per-range execution and result assembly testable without constructing a
  `FitManager`.
- One home for constraint behaviour: `fitting/constraints.py`.
- `fit()`, `fit_folded()`, `fit_frange()` and every property keep their
  current signature and behaviour.

## Non-goals

- Redesigning the fit pipeline (QEP-070, Implemented).
- Changing fit results numerically. The parity suite must not move.
- Splitting `result.py`, `models.py`, `folding.py` or `guess.py` (see
  Follow-ups).
- Changing `FreqCutoff`, `ParameterGuesser` or the backend seam.

## Decisions

### D1 -- Extract collaborators and functions, not mixins

**Context.** The groups above could be moved either as mixin classes that
`FitManager` inherits, or as separate objects and module functions.

**Decision.** Collaborators and functions. `RangeFitter` is constructed by
`FitManager.__init__` and holds the backend, options, cutoff and constraint
manager; input preparation and result assembly become module-level functions
taking explicit arguments.

**Consequences.** Dependencies become visible in signatures, and each piece is
testable alone. `FitManager` keeps a small amount of forwarding code.

**Rejected:** mixins. They would cut the line count without reducing coupling
-- every mixin would still read `self._backend`, `self._model` and
`self._freq_cutoff`, so the class stays one object split across files, and
`architecture.md` prefers composition over deep inheritance.

### D2 -- `constraints` returns the `ConstraintManager` (breaking)

**Context.** Five public methods on `FitManager` forward to
`ConstraintManager` with the same guard. `fm.constraints` currently returns
`dict[str, Constraint]`.

**Decision.** `fm.constraints` returns the `ConstraintManager`, which gains a
`Mapping[str, Constraint]` interface (`__getitem__`, `__iter__`, `__len__`,
`items()`) plus `set()`, `set_free()`, `to_array(n_pixel)` and `types()`. The
five delegators are removed. The contrast fan-out moves into
`ConstraintManager.set()`.

**Consequences.** Read access is unchanged: `fm.constraints["center"].vmin`
still works, so most tests and all recorded-metadata code are untouched.
Mutating calls change:

| Before | After |
|---|---|
| `fm.set_constraints("center", vmin=2.8)` | `fm.constraints.set("center", vmin=2.8)` |
| `fm.set_free_constraints()` | `fm.constraints.set_free()` |
| `fm.get_constraints_array(n)` | `fm.constraints.to_array(n)` |
| `fm.get_constraint_types()` | `fm.constraints.types()` |

`FitManager(constraints={...})` is unchanged, which is how `Measurement`,
`measurement_workflows` and the GUI pass constraints, so no GUI change is
required. Accessing `fm.constraints` before the model is resolved keeps
raising `ModelNotResolvedError`.

**Rejected:** keeping the delegators. The user chose the break: pre-1.0, few
users, and keeping them means the duplication this QEP exists to remove.

### D3 -- `_param_idx` moves with the constraints

`_param_idx` maps a parameter type name (and the deprecated aliases) to
parameter indices. It is used by the constraint machinery and by
`_mt_center_window_for_range`. It becomes a function in `constraints.py`
taking `(model, parameter)`, keeping the `DeprecationWarning` for aliases.

## Proposed design

```
fitting/
  manager.py         FitManager: construction, fit(), fit_folded(),
                     model resolution, guessing, properties         (~350)
  fit_inputs.py      PreparedFitInputs, validate_inputs(),
                     prepare_data(), freq-cutoff validation         (~150)
  range_fit.py       RangeFitter: per-range constraints, cutoff,
                     backend execution, reshaping                   (~330)
  result_assembly.py assemble_fit_result(): parameter dict, quality
                     metrics, metadata (incl. fit_backend/estimator
                     provenance and fit_constraints/fit_freq_cutoff) (~120)
  constraints.py     Constraint, ConstraintOverride, ConstraintManager
                     (now Mapping + set/set_free/to_array/types),
                     param_idx()                                    (~380)
```

Dependency direction stays downward: `manager -> range_fit -> backends`,
`manager -> result_assembly -> result`, everything -> `constraints`. No new
module imports `manager`.

`RangeFitter` sketch:

```python
class RangeFitter:
    def __init__(self, backend, options, freq_cutoff, settings): ...
    def fit_range(self, model, data, freq, initial, *, irange, n_frange,
                  constraints, overrides=None) -> _RangeFitOutputs: ...
    def fit_all(self, model, prepared, initial, constraints,
                overrides=None) -> list[_RangeFitOutputs]: ...
```

`FitManager.fit_frange()` stays as the public entry point `refit.py` uses and
delegates to `RangeFitter.fit_range()`.

## Implementation steps

Each phase ends with `uv run pytest`, `uv run ruff check .`,
`uv run ruff format --check .` and `uv run ty check src/qdmpy` clean.

- **Phase 0 -- baseline.** Record `tests/test_fit_pipeline_parity.py` results
  and a fit on `reference_data/` NPZ fixtures; these must not change in any
  later phase.
- **Phase 1 -- `constraints.py`.** Add the Mapping interface, `set()`,
  `set_free()`, `to_array()`, `types()` and `param_idx()`; move the contrast
  fan-out in. Remove the five `FitManager` methods and update callers and
  tests (`test_fit.py`, `test_refit.py`, `test_fit_pipeline_parity.py`).
  Update `docs/tutorials/fitting.md` and `docs/extending.md`.
- **Phase 2 -- `fit_inputs.py`.** Move `_PreparedFitInputs` and the
  validation/preparation functions; `fit()` calls them.
- **Phase 3 -- `result_assembly.py`.** Move `_assemble_result` to a function
  taking `(model, prepared, outputs, pixel_spacing, metadata_inputs)`.
- **Phase 4 -- `range_fit.py`.** Move per-range execution into `RangeFitter`;
  `FitManager` constructs one and forwards `fit_frange`.
- **Phase 5 -- docs and guardrails.** Update `memory/architecture.md` and
  `memory/fitting.md` with the new module graph; add the module sizes to the
  review checklist; CHANGELOG entry.

## Risks and tradeoffs

- **Silent numerical drift while moving code.** Mitigation: Phase 0 baseline
  plus the existing parity suite, checked after every phase.
- **Circular imports.** `constraints.py` must not import `manager`;
  `param_idx()` takes a `Model`, not a `FitManager`.
- **Churn against QEP-FIT-001 / QEP-FIT-005**, which touch `result.py` and
  `models.py`. Neither file is edited here beyond imports.
- **More files to navigate.** Accepted: the modules are cohesive and named
  after what they do, per `architecture.md` (no `utils.py` catch-alls).

## Acceptance criteria

- `wc -l src/qdmpy/fitting/manager.py` < 600; no new module > 400 lines.
- `uv run pytest` passes with the same test count (plus new unit tests), and
  `test_fit_pipeline_parity.py` is unchanged except for the constraint API
  rename.
- Fitting `reference_data/` fixtures gives bit-identical parameters to the
  Phase 0 baseline.
- `RangeFitter` and `assemble_fit_result()` have direct unit tests that do not
  construct a `FitManager`.
- `fit()`, `fit_folded()`, `fit_frange()` signatures unchanged; `refit.py` and
  `measurement_workflows.py` untouched apart from imports.
- qdmpy-gui test suite passes with no GUI source change.

## GUI Integration Requirements

1. **Touchpoints.** The GUI uses `Measurement.fit_odmr` /
   `fit_folded_odmr` / `refit_outliers` and passes `constraints=` dicts;
   `app/fit_constraints_vm.py` imports
   `qdmpy.fitting.constraints._mt_to_absolute_ghz`. It never calls
   `FitManager.set_constraints` or the other removed delegators.
2. **Persisted data.** None. `.qdm` contents and
   `FitResult.metadata['fit_constraints']` keep their current shape.
3. **User-facing behaviour.** No change: no new progress, warning or error
   paths.
4. **Acceptance.** GUI suite passes unchanged; manual check of load -> fit
   (with mT constraints set in the dialog) -> refit -> save/reload.
5. If `_mt_to_absolute_ghz` is renamed during Phase 1, it must be exported
   under a public name and the GUI import updated in the same change.

## Open questions

- Should `fit_frange` stay on `FitManager`, or should `refit.py` take a
  `RangeFitter` directly? Keeping it is the smaller change; moving it would
  let a refit run without a `FitManager`. Deferred until Phase 4, when the
  seam is visible.

## Follow-ups (out of scope)

| File | Lines | Limit | Natural split |
|---|---|---|---|
| `fitting/result.py` | 809 | 800 | persistence (`save_results`/`load_results`) vs derived-quantity properties; QEP-FIT-001 already reworks this file |
| `fitting/models.py` | 807 | 800 | line-shape functions and Jacobians vs `Model`/`ModelRegistry`; QEP-FIT-005 removes `_main_demo` (~10 lines) |
| `odmr/folding.py` | 748 | 600 (guideline) | D_ZFS search vs folding/residual computation |
| `fitting/guess.py` | 625 | 600 (guideline) | numba kernels vs the guess API |
