# QEP-068 - Fit Backend Seam (GPU / CPU / Fake)

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | QEP-029 (config/execution split), QEP-059 (constraint unification) |
| **Blocks** | QEP-FIT-003, QEP-060 (both become easier to verify) |
| **Author** | QDMpy Team |
| **Created** | 2026-07-04 |

---

## Motivation

The GPU dependency is not a seam — it is a scattered hard conditional.

1. **Availability is checked in six places.** `is_pygpufit_available()` /
   `self._gpu_available` guards live in `fitting/manager.py:80`, `:290`, `:619`,
   `:708`, `measurement_workflows.py:166-176`, and `:327-332`, each raising its
   own copy of the same `DependencyError` message.

2. **A `gpu_available: bool` flag is hand-threaded through every fit-adjacent
   signature**: `Measurement.fit_odmr`, `fit_folded_odmr`, `refit_outliers`,
   `_validate_fit_prerequisites`, plus four workflow functions and
   `FitManager.__init__`. Adding any backend-related behavior means editing all
   of them in lockstep.

3. **The documented custom-model contract is dead.** `Model` documents
   `self.model_id = -1  # CPU-only; gpufit not used for custom models`
   (`fitting/models.py:206`), but there is no CPU path: `fit_frange` calls
   `pygpufit.gpufit.fit_constrained(model_id=model.model_id)` unconditionally
   (`manager.py:640-651`). A registered pure-Python model crashes inside
   gpufit. `Model.func()` — the CPU evaluator every model must implement — is
   used only for residual logging (`guesser.py:185`) and plotting, never for
   fitting.

4. **Tests pay the price.** Because the GPU call has no interface, fit-adjacent
   suites assert wiring through monkeypatching: 46 mock/patch sites in
   `tests/test_measurement.py`, 28 in `tests/test_refit.py`, 18 in
   `tests/test_load.py`, 11 in `tests/test_folded_fit.py`. Real behavior is
   only exercised in GPU-gated integration tests.

A single narrow `FitBackend` interface with two real adapters (gpufit, scipy)
and one test fake turns all of this into ordinary dependency injection.

---

## Goals

- One place in the codebase decides "can we fit, and with what".
- `FitManager` depends on a `FitBackend` abstraction, never on `pygpufit`
  directly.
- Pure-Python custom models become fittable (CPU backend), honoring the
  contract already documented on `Model`.
- Fit-adjacent unit tests run without a GPU and without monkeypatching
  internals, via an injectable fake backend.
- Delete the `gpu_available` parameter from all public signatures after a
  deprecation window.

## Non-goals

- No change to fit results, physics conventions, units, or the GHz-only
  frequency contract.
- No performance parity between CPU and GPU backends: the scipy backend is for
  custom models, small ROIs, and CI — not 2k×2k production frames.
- No decomposition of `FitManager.fit()` itself (QEP-FIT-003) and no folded/
  non-folded path unification (QEP-060). This QEP only inserts the seam those
  QEPs will benefit from.
- No new fitting algorithms beyond a least-squares CPU reference.

---

## Design

### New module: `fitting/backends.py`

```python
@dataclass(frozen=True)
class BackendFitOutput:
    """Raw per-fit output, shape-flat: one row per (pol, pixel) fit."""

    parameters: NDArray   # (n_fits, n_params) float32
    states: NDArray       # (n_fits,) int32 — 0 == converged (gpufit convention)
    chi2: NDArray         # (n_fits,) float32
    iterations: NDArray   # (n_fits,) int32
    execution_time: float # seconds


class FitBackend(Protocol):
    """Everything FitManager needs to know about an optimizer."""

    name: str

    def is_available(self) -> bool: ...

    def supports(self, model: Model) -> bool: ...

    def fit(
        self,
        data: NDArray,                 # (n_fits, n_freq) float32
        freq_ghz: NDArray,             # (n_freq,) float64
        initial_parameters: NDArray,   # (n_fits, n_params) float32
        constraints: NDArray,          # (n_fits, 2 * n_params) float32
        constraint_types: NDArray,     # (n_params,) int32
        model: Model,
        options: FitBackendOptions,    # estimator, max_iterations, tolerance
    ) -> BackendFitOutput: ...
```

`FitBackendOptions` is a small frozen model carrying `estimator`,
`max_number_iterations`, `tolerance` — the three values currently read from
`settings.fit` inside `fit_frange`.

### Adapters

**`GpufitBackend`** — the current body of `FitManager.fit_frange`
(`manager.py:623-652`) moves here verbatim, including the GHz/AHYP comment.
`is_available()` absorbs today's `is_pygpufit_available()`;
`supports(model)` returns `model.model_id >= 0`.

**`ScipyBackend`** — per-pixel `scipy.optimize.least_squares` over
`Model.func` (scipy is already a runtime dependency via peak detection).
`LOWER`/`UPPER`/`LOWER_UPPER` constraint types map to `least_squares` bounds;
`FREE` maps to ±inf. `supports(model)` returns `True` for any model.
Estimator: LSE only; requesting MLE logs a warning and falls back to LSE.
Output states follow the gpufit convention (0 = converged) so downstream
quality metrics are unchanged.

**`FakeFitBackend`** — lives in `qdmpy.testing` next to the existing
`make_synthetic_*` helpers. Returns the initial parameters unchanged with
`states=0`, `chi2=0`. Exists so `Measurement`/`refit`/workflow tests assert
behavior instead of patching `pygpufit` import paths.

### Resolution

```python
def resolve_backend(spec: FitBackend | str = 'auto') -> FitBackend:
    """'auto' → gpufit if available, else raise DependencyError with a hint
    to pass backend='scipy' explicitly. Never silently falls back to CPU."""
```

`'auto'` deliberately does **not** fall back to scipy: a silent CPU fallback
would turn a seconds-long GPU fit of a 2k×2k frame into hours. The error
message names the explicit opt-in (`backend='scipy'`). Accepted specs:
`'auto'`, `'gpufit'`, `'scipy'`, or any `FitBackend` instance.

A new settings knob provides the default:

```python
class FitSettings(BaseModel):
    backend: Literal['auto', 'gpufit', 'scipy'] = 'auto'
    ...
```

### FitManager changes

```python
FitManager(
    model_name='ESR14N',
    constraints=None,
    *,
    freq_cutoff=None,
    settings=None,
    backend: FitBackend | str | None = None,   # new; None → settings.fit.backend
    gpu_available: bool | None = None,          # deprecated, see Migration
)
```

- `fit_frange` becomes a thin shape adapter over `self._backend.fit(...)`.
  The refit path (`fitting/refit.py:304` calls `fit_manager.fit_frange`)
  inherits the seam with zero changes.
- If `self._backend.supports(model)` is false at fit time, raise
  `DependencyError` naming the model, the backend, and the fix
  (`backend='scipy'`).
- The four availability checks inside `manager.py` collapse into backend
  resolution at construction time.

### Call-site cleanup

- `measurement_workflows.validate_processed_odmr` keeps its processed-data
  validation but loses the GPU check; `fit_folded_measurement_odmr` loses its
  duplicated `DependencyError` block.
- `Measurement.fit_odmr` / `fit_folded_odmr` / `refit_outliers` gain
  `backend: FitBackend | str | None = None`, forwarded as one value.
- `is_pygpufit_available()` remains exported from `qdmpy` (public API) and
  delegates to `GpufitBackend().is_available()`.

---

## Migration Plan

**Phase 1 — seam behind the current surface.**
Add `fitting/backends.py` with `GpufitBackend` + `FakeFitBackend` +
`resolve_backend`. `FitManager` consumes a backend internally; all public
signatures unchanged. Full test suite must be green with zero behavioral
diff (gpufit adapter is a code move, not a rewrite).

**Phase 2 — thread `backend`, deprecate `gpu_available`.**
Add the `backend` parameter to `FitManager`, `Measurement` methods, and
workflows. `gpu_available` emits `DeprecationWarning` and maps:
`True` → treat gpufit as available (today's test override), `False` → raise
`DependencyError` (today's behavior). Migrate the mock-heavy tests in
`test_measurement.py` / `test_refit.py` / `test_folded_fit.py` to
`FakeFitBackend`; mock-site count is the tracked metric (expect ≥ 60 of ~85
sites deleted).

**Phase 3 — ScipyBackend.**
Implement, plus `supports()` dispatch and the `fit.backend` settings knob.
Add a CPU-vs-GPU consistency test on a small synthetic frame (tolerances, not
bit-equality) and a fittable pure-Python custom-model example in the
extension docs — replacing the currently false `model_id = -1` promise.

**Phase 4 — removal.**
Delete `gpu_available` parameters in the next minor release, following the
deprecation policy defined by QEP-057.

Each phase runs `uv run pytest` and notes the failure-count delta per the QEP
workflow.

---

## Alternatives Considered

1. **Add a CPU branch inside `fit_frange`** (`if not gpu: _fit_cpu(...)`).
   Rejected: keeps the scattered conditionals, adds a flag-argument branch
   (banned by house style), and still provides no test seam.

2. **Subclass FitManager (`GpuFitManager` / `CpuFitManager`).**
   Rejected: undoes the QEP-029 configuration/execution split, forces callers
   to choose a class instead of a value, and duplicates the pipeline the
   moment it diverges — the exact drift QEP-060 exists to prevent.

3. **Require Gpufit's CPU build.**
   Rejected: no maintained pip distribution, high install burden for exactly
   the users (CI, laptops, custom models) the CPU path serves, and still no
   injectable fake for tests.

4. **Silent auto-fallback to scipy when gpufit is missing.**
   Rejected: turns an install problem into an hours-long surprise on
   production-sized frames. Explicit opt-in keeps the failure loud and the
   fix one keyword away.

---

## GUI Integration Requirements

1. **Core API/data contract touchpoints:** `qdmpy-gui` calls
   `Measurement.fit_odmr` / `fit_folded_odmr` / `refit_outliers` and reads
   `QDMResult`. Result shape, parameter names, and quality metrics are
   unchanged. The only surface additions are the optional `backend` keyword
   and the `fit.backend` settings key; `gpu_available` keeps working through
   Phase 3 (with `DeprecationWarning`).
2. **Settings/migration:** new `fit.backend` key defaults to `'auto'`, which
   reproduces today's behavior exactly; no migration of persisted GUI
   settings is required. If the GUI later surfaces a backend selector, it
   binds to this key.
3. **Error/progress behavior:** when gpufit is missing, the GUI now receives
   one `DependencyError` (raised at backend resolution) whose message names
   `backend='scipy'` as the remedy, instead of six differently-located copies
   of the old message. Scipy-backend fits are slow; the GUI should treat them
   like any long fit (existing progress/busy handling suffices — no new
   progress API is added by this QEP).
4. **Acceptance checks:** with gpufit installed: load → fit → inspect maps →
   save `.qdm` → reload, byte-identical parameter arrays vs. pre-QEP build.
   Without gpufit: fit attempt surfaces the single `DependencyError` cleanly
   in the GUI (no traceback dialog); explicitly selecting the scipy backend
   on a small ROI completes and renders maps. No GUI-only workaround needed.
5. **Regression rationale:** phases 1–2 are internal moves; a GUI smoke test
   (load → fit → display) against a Phase-2 build is sufficient to confirm no
   regression.

---

## Testing Plan

- **Unit:** `resolve_backend` matrix ('auto'/'gpufit'/'scipy'/instance ×
  gpufit present/absent); `GpufitBackend.supports` on `model_id = -1`;
  constraint-type → bounds mapping in `ScipyBackend`; MLE→LSE fallback
  warning.
- **Behavioral:** `Measurement.fit_odmr` end-to-end with `FakeFitBackend`
  (no GPU, no mocks) asserting result assembly, metadata, and refit routing.
- **Consistency:** ScipyBackend vs GpufitBackend on a small synthetic ESR14N
  frame within tolerance (GPU-gated, lives beside
  `tests/integration/test_gpufit_consistency.py`).
- **Regression:** existing reference-data regression suite unchanged and
  green after every phase.

## Success Criteria

- `pygpufit` is imported in exactly one module (`fitting/backends.py`).
- Zero `gpu_available` parameters on public signatures at Phase 4.
- A pure-Python registered model fits successfully via `backend='scipy'`.
- Mock/patch sites across the four fit-adjacent test files reduced by ≥ 70%.
