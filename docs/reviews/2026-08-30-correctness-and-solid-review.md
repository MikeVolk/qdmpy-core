# Correctness & SOLID Review -- qdmpy-core (redo)

| Field       | Value                                                          |
|-------------|-----------------------------------------------------------------|
| Date        | 2026-08-30                                                     |
| Branch      | `develop` @ `4e3c88d`                                           |
| Scope       | `src/qdmpy/` (~12.9k LOC) full read, + `alignment/` on unmerged `feature/qep-067-alignment-stitching` |
| Suite state | 1047 passed, 89.71% coverage                                    |
| Lint state  | `ruff check src/ tests/` clean; `ty check src/qdmpy` clean (3 stale `# ty: ignore` comments, no real diagnostics) |

This redoes a review request that was originally typed into the wrong working
directory (`related_repos/Gpufit`, 2026-08-29) and reviewed that CUDA fork
instead of this repo -- that review stands on its own and is not repeated
here. This one targets `qdmpy-core` as intended.

There is a prior review at
[`2026-08-22-math-and-architecture-review.md`](2026-08-22-math-and-architecture-review.md)
covering math correctness in depth. Since then `develop` absorbed five QEPs
touching the fitting pipeline directly (QEP-068 FitBackend seam, QEP-069
TorchBackend, QEP-070 pipeline unification, QEP-071 refit consistency,
QEP-073 analytic Jacobians), so this pass concentrated fresh review effort
there and on the modules the prior review touched more lightly for SOLID
(`odmr/`, `io/`, `plotting/`, `cli/`, top-level orchestration). Numeric and
behavioral claims below were verified by execution (repro scripts under
`/tmp`, or direct function calls), not asserted from reading.

`alignment/` does not exist on `develop` -- it lives on unmerged branch
`feature/qep-067-alignment-stitching`, 19 commits behind `develop`, reviewed
via a `git worktree`. Its findings (Part 3) are flagged separately since they
don't affect the current `develop` suite state above.

---

## Part 0 -- Status of the 2026-08-22 findings, spot-checked

Not a full re-verification of every item in the prior review -- only the ones
the QEP-068..073 churn plausibly touched:

- **F11 (`FitManager` self-mutation during `fit()`) -- FIXED.** Verified by
  execution: `FitManager.constraints["center"]` is bit-identical before and
  after a two-range mT-mode `fit()` call, post QEP-070 phase 2.
- **F3 (refit bypasses `freq_cutoff`/constraint preprocessing) -- FIXED.**
  QEP-071's target. Verified both `_fit_all_franges()` and `fit_frange()` now
  compute `_effective_constraints_for_range` from the same pre-cutoff
  frequency array before `_apply_freq_cutoff_for_range`.
- **F2 (refit writes back unconditionally, even if worse) -- STILL OPEN.**
  QEP-071 fixed *what* the refit fit uses (F3) but never added the
  accept/reject-on-chi2 guard the prior review flagged as the single
  highest-value fix in the file. Re-verified with a fresh repro -- see Part 1
  finding 1 below.
- F1, F4-F10, F12, F13, A1, A2, A4-A7 were not re-checked this pass; assume
  still open unless independently reproduced.

---

## Part 1 -- `fitting/` (QEP-068/069/070/071/073 churn zone)

`guess.py`, `guesser.py`, `result.py` had zero diff since 2026-08-22; all
churn is in `backends.py`, `constraints.py`, `freq_cutoff.py`, `manager.py`,
`models.py`, `refit.py`, `torch_backend.py`.

### Bugs

**1. `refit.py:322-326` (`_refit_pass`) -- HIGH, carried over as F2, still open.**

Refit writes the new fit back with no comparison to the pixel's pre-refit
chi2. No test in `tests/test_refit.py` covers a chi2 regression. Verified:
constructed an outlier pixel at chi2=5.0 with a fake backend that always
returns chi2=999.0 -- `refit_outliers()` overwrote it unconditionally:

```
outlier pixel BEFORE: chi2=5.0, center=2.869999885559082
outlier pixel AFTER:  chi2=999.0, center=3.369999885559082
Refit wrote back a WORSE fit (chi2 999 > 5.0): True
```

Fix: compare `new_chi2_arr` against the pixel's existing chi2 in
`_refit_pass` and only overwrite where the refit improved.

**2. `backends.py:251-269` (`ScipyBackend._fit_all_pixels`) -- HIGH, new in QEP-068.**

No NaN guard. `scipy.optimize.least_squares` raises a bare `ValueError` on
the first NaN-containing pixel, which propagates uncaught through
`FitManager.fit()`/`fit_frange()` and aborts the fit for *every* pixel in the
call, not just the bad one. `TorchBackend` handles identical input
gracefully -- it marks the pixel `states=2` (INVALID) via
`isfinite(chi2) & isfinite(y).all(-1)` and returns the initial guess, leaving
every other pixel to fit normally. Verified with one NaN frequency point
injected into an otherwise-clean ESR14N spectrum:

```
scipy raised: ValueError('Residuals are not finite in the initial point.')
torch params: [2.875 0.005 0.05  0.09  0.04  0. ] states [2] chi2 [0.08047761]
```

NaN pixels are reachable in production: dead/saturated camera pixels,
`HotPixelFilter(replacement='nan')`, or the refit path (reads `data.values`
directly, bypassing the guesser). Fix: wrap the per-pixel `least_squares`
call (or pre-check `np.isfinite(target).all()`) and mark that pixel invalid
like `TorchBackend` does, instead of raising.

### SOLID

**3. LSP -- `FitBackend` Protocol has no documented non-finite-input contract; implementations diverge.**

Same evidence as bug 2. `backends.py:64-92` specifies shapes/types but not
NaN semantics, so `GpufitBackend`/`ScipyBackend`/`TorchBackend` are not
drop-in replacements on real (imperfect) data -- code that works on
`backend='torch'` can crash outright after switching to `backend='scipy'`.
Fix: document the contract on the Protocol explicitly and bring
`ScipyBackend` into compliance.

**4. DRY (minor) -- `backends.py:286-325` and `torch_backend.py:330-376` duplicate the "probe `model.jacobian`, validate shape, fall back to finite differences" skeleton.**

Shapes genuinely differ per framework (numpy `(1, n_freq)` vs torch
`(n_fits, n_freq)` probes), so a full merge isn't free, but the
validate-and-fallback skeleton could be one framework-parameterized helper
(arguably belongs on `models.py`, since it validates the `Model.jacobian`
contract, not backend-specific logic).

**5. Minor -- `manager.py:205-210` (`_resolve_backend`)'s deprecated `gpu_available` path constructs `GpufitBackend` directly instead of going through `resolve_backend()`/`_BACKENDS_BY_NAME`,** unlike every other backend-resolution path. Acceptable as isolated, explicitly-deprecated compatibility code; worth removing alongside `gpu_available` rather than leaving a second resolution mechanism live.

### Verified correct (no bug)

`esr14n_jacobian`/`esr15n_jacobian`/`esrsingle_jacobian` (QEP-073) checked
against finite differences for all three models -- all six-parameter columns
matched to expected FD truncation error, not analytic error.
`ScipyBackend`/`TorchBackend(cpu)` end-to-end fits (unconstrained,
box-constrained including a true-center-outside-window clamp case, and
`n_pixel=1`) converge to matching, correct parameters between backends.

---

## Part 2 -- `odmr/` + `io/`

### Bugs

**1. `odmr/data.py:51-67` -- HIGH -- `ODMRData.validate_data_array` never validates the `freq_ghz` coordinate on the real ingestion path.**

`validate_frequencies()` is only called from `ODMRData.from_numpy()`.
`ODMRData.from_loader()` -- the path every real loader (`MatlabLoader`) goes
through -- constructs `ODMRData(data=...)` directly, which checks dims/dtype
/coord-presence but not values. Verified: a `freq_ghz` coord with a NaN and
non-monotonic values constructs silently via `ODMRData(data=da)`; the
equivalent through `from_numpy()` raises `DataValidationError`. Fix: call
`validate_frequencies(v.coords["freq_ghz"].values)` inside the shared
field_validator so both construction paths share one contract.

**2. `odmr/processors.py:160-171` -- MEDIUM -- `BinningProcessor.process()` silently produces a zero-sized array when `bin_factor` exceeds the scan dimensions.**

`xr.DataArray.coarsen(..., boundary="trim")` trims to zero rows/cols with no
error, and `ODMRData`'s validator checks dim names only, not sizes. Verified:
`BinningProcessor(bin_factor=8)` on a 4x4 scan produces shape
`(2, 2, 0, 0, 20)`, `scan_dimensions=(0, 0)`, and constructs as valid
`ODMRData`. Fix: validate `bin_factor <= min(ny, nx)` in `process()`, and/or
a min-size check in `ODMRData`'s validator.

**3. `odmr/processors.py:137-147` -- LOW -- `NormalizationProcessor` divides by an unguarded per-pixel mean/max,** producing silent NaN for an all-zero pixel spectrum (e.g. a dead sensor pixel) with no warning.

**4. `io/qdm.py:208-222` -- LOW -- `_check_version` is asymmetric,** rejecting files newer than the code understands but never files declaring an older schema baseline. Harmless today (only `"1.0"` exists) but the guard should be `major != code_major` or an explicit migration table rather than one-directional, before a schema-breaking bump lands.

### SOLID

**5. HIGH -- Open/Closed violation: `odmr/processors.py:218-225` (`ProcessorSpec`) and `:332-338` (`ODMRProcessorManager.from_config`) hardcode the four built-in processor types in a discriminated Union, contradicting the `Processor` protocol's own documented custom-processor contract** (processors.py:23-55, which advertises that user processors need not inherit from any base class). A custom processor works fine through `add_processor()`/`process()`, but round-tripping through `to_config()`/`from_config()` fails hard with a pydantic tag-validation error. This is the exact anti-pattern `architecture.md` warns against for ESR models. Fix: a `ProcessorRegistry` dict keyed by `type` string (mirroring `ModelRegistry.register`), with `from_config` dispatching through it instead of a closed Union.

**6. MEDIUM -- Single Responsibility violation: `odmr/io.py:65-169` (`MatlabLoader.load()`) is a ~100-line method doing file discovery, mat73/scipy parsing, image-stack-layout detection, frequency parsing, polarity stacking, spatial reshaping, and y-axis flipping,** complexity-suppressed via `# noqa: C901, PLR0912, PLR0915`. Contradicts the project's own stated limits. Fix sketch: extract `_load_single_file`, `_parse_frequencies`, `_assemble_data_array` as separate methods; `load()` becomes a ~15-line orchestrator.

---

## Part 3 -- `alignment/` (unmerged `feature/qep-067-alignment-stitching`), `plotting/`, `cli/`, top-level

### Bugs -- on `develop`

**1. `utils.py:160,164` (`double_norm`) -- CRITICAL -- crashes on its own documented default; broken for any non-last axis.**

`np.expand_dims(np.min(result, axis=axis), data.ndim - 1)` always inserts the
restored dimension at the *last* axis position, not at `axis`. Verified:
`double_norm(arr_2d, axis=None)` (the documented "normalize globally"
default) raises `AxisError` outright; `double_norm(arr_2d, axis=0)` raises a
broadcast `ValueError`. Only `axis == data.ndim - 1` happens to work, which
is why the sole existing test (1D array, default axis) never caught it. The
function is public API (`qdmpy.plotting.__all__`). Fix: keepdims-style
expansion at `axis`, not at `data.ndim - 1`.

**2. `utils.py:36-41` (`millify`) -- HIGH -- wrong clamp bounds; crashes for large values, silently truncates small ones.**

`millidx` is clamped to `[0, 7]` but then indexed as
`MILLNAMES[millidx + 3]`, valid only for `millidx in [-3, 4]`. Verified:
`millify(1e15)` (and 1e18, 1e21, 1e24) raises `IndexError`;
`millify(0.0005)` and `millify(1e-8)` silently return `"0.0"` instead of a
milli-prefixed value. Fix: clamp to `[-3, 4]` before the offset.

**3. `plotting/odmr.py:396` (`plot_fluorescence_correction`) -- HIGH -- polarity sign labels are inverted.**

`polarity_label = {0: "+", 1: "-"}.get(p, ...)`, but index 0 is `neg` and
index 1 is `pos` (per CLAUDE.md and confirmed via
`make_synthetic_odmr_data()`'s `polarity` coord). Every fluorescence-correction
preview plot mislabels which polarity is which sign. Fix:
`{0: "-", 1: "+"}`, or better, read the actual `polarity` coord string.

**4. `cli/__init__.py:30` (`main`) -- MEDIUM -- CLI version always reports "unknown" on a real install.**

`get_version("QDMpy")`, but `pyproject.toml` declares `name = "qdmpy-core"`.
Verified with the stale, gitignored dev-only `src/QDMpy.egg-info` removed:
`version("QDMpy")` raises `PackageNotFoundError`. Fix:
`get_version("qdmpy-core")`.

**5. `magnetic_map.py:249-265` (`MagneticMap.display`) -- MEDIUM -- documented kwarg passthrough crashes.**

Signature/docstring advertise `**imshow_kwargs` forwarded to
`.plot(**imshow_kwargs)`, but it forwards to
`plot_magnetic_component(self, component, **imshow_kwargs)`
(`plotting/fields.py`), whose signature has no `**kwargs`. Verified:
`mm.display("Bz", cmap="plasma")` raises `TypeError`. Fix: add `**kwargs` to
`plot_magnetic_component` and thread it to `ax.imshow`, or drop the dead
parameter.

### Bugs -- on `feature/qep-067-alignment-stitching` (not yet on `develop`)

**6. `alignment/_registration.py:160-181` (`_score_registration`) -- HIGH -- the `error` param from `phase_cross_correlation` is dead code; it never affects the returned score.**

Both branches of the `isfinite(error)` check return the same
already-clamped `corr_score`. Verified: `error=1.0`, `error=np.inf`,
`error=np.nan` with identical `ref`/`moving`/`shift` all return the same
score. The intended quality signal from `phase_cross_correlation`'s
registration error is silently discarded.

**7. `alignment/_registration.py:195-222` (`_fuse_channel_transforms`) -- HIGH -- no outlier rejection across channels; `confidence` is a mean-quality score, not an agreement metric.**

QEP-067's own proposal (`proposals/QEP-067-measurement-alignment.md:93,260`)
promises RANSAC outlier rejection when fitting the affine matrix across
channels; the shipped code does a pure per-channel weighted average with
none. Verified: two channels agreeing at shift `(3,3)` and one diverging
wildly at `(50,50)`, all with score 0.9, fuse to translation `(18.67,
18.67)` -- nowhere near either estimate -- with `confidence=0.9`, well above
the `_LOW_CONFIDENCE_THRESHOLD = 0.5` gate that exists to catch exactly this.

### SOLID -- on `feature/qep-067-alignment-stitching`

**8. HIGH -- Dependency Inversion / Open-Closed violation: `plotting/display.py:12` and `plotting/fit.py:11` module-level import `qdmpy.alignment.Mosaic` and branch on `isinstance(result, Mosaic)` in four-plus functions each.**

Same anti-pattern `architecture.md` calls out for models, applied to
result-container types: every future result-like type requires editing every
one of these functions. It also means all consumers of `qdmpy.plotting`
transitively import scikit-image at module-import time (compounds the prior
review's A3 import-cost finding). Fix: a narrow `Protocol` (e.g.
`HasStitchedFields`) that `Mosaic`/`FitResult`/`QDMResult` all structurally
satisfy, dispatched via `hasattr`/protocol conformance instead of a concrete
`isinstance` import.

**9. LOW -- `alignment/_mosaic.py:34,46-48` -- `Mosaic.bxyz` is declared and lock-protected but never populated;** `stitch()` never sets it. Either wire it up or drop the field until there's a producer.

---

## Part 4 -- Recommended order of work

| # | Finding | Change | Size |
|---|---------|--------|------|
| 1 | Part 1.1 (F2) | Chi2-guarded write-back in `refit._refit_pass` | 3 lines -- still the highest-value fix in the file, unchanged since 2026-08-22 |
| 2 | Part 3.1 | `double_norm` axis bug -- crashes on its documented default, public API | small |
| 3 | Part 2.5 | `ProcessorRegistry` for OCP-compliant `from_config` | small-medium |
| 4 | Part 1.2 / 1.3 | `ScipyBackend` NaN handling to match `TorchBackend`; document the `FitBackend` contract | small |
| 5 | Part 3.3 | Fix inverted polarity labels in `plot_fluorescence_correction` | 1 line |
| 6 | Part 3.2 | `millify` clamp bounds | small |
| 7 | Part 2.1 | `validate_frequencies` on the `from_loader` construction path | small |
| 8 | Part 2.2, 2.3 | `BinningProcessor` size guard; `NormalizationProcessor` zero-guard | small |
| 9 | Part 3.4, 3.5 | CLI version string; `MagneticMap.display` kwarg passthrough | small |
| 10 | Part 2.6 | Decompose `MatlabLoader.load()` | medium |
| 11 | Part 3.6, 3.7, 3.8 (alignment branch) | Wire up `_score_registration`'s error signal; add cross-channel outlier rejection per QEP-067's own spec; Protocol-ize the `Mosaic` isinstance branching -- resolve before merging `feature/qep-067-alignment-stitching` | medium, pre-merge blocker for that branch |
