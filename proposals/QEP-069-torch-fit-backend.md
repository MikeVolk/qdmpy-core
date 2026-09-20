# QEP-069 - Torch Fit Backend (Architecture-Independent GPU Fitting)

| Field | Value |
|-------|-------|
| **Status** | Implemented (2026-07-04) |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | QEP-068 (fit backend seam) |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-07-04 |

---

## Motivation

QEP-068 created the `FitBackend` seam, but the two adapters it shipped leave a
gap in practice:

1. **`GpufitBackend` is CUDA-only.** pyGpufit is distributed as local
   platform wheels whose native library requires an NVIDIA GPU; on Apple
   silicon the bundled Linux `.so` cannot even be loaded. Users without a
   CUDA machine cannot fit at production scale at all.
2. **`ScipyBackend` cannot reach production scale.** Per-pixel
   `scipy.optimize.least_squares` on a typical frame — 1200 x 1920 px x
   2 polarities ≈ 4.6 million fits per frequency range — takes hours. It
   exists for custom models, tiny ROIs, and CI, not real data.

This QEP adds a third adapter, `TorchBackend`: a batched Levenberg-Marquardt
optimizer written in PyTorch that runs the *same* code path on NVIDIA CUDA,
Apple-silicon MPS, and CPU. All fits in a chunk are advanced in parallel as a
batch dimension, so a full frame is a handful of large tensor ops per LM
iteration instead of millions of Python-level optimizer calls.

A key enabler is already in the codebase: the ESR model functions
(`esr14n`/`esr15n`/`esrsingle`) are pure arithmetic and broadcasting — their
only numpy call is `np.atleast_2d`. Making that single call framework-neutral
lets the identical model code evaluate torch tensors on the GPU. No physics
is duplicated, and custom models written in the same style are GPU-fittable
automatically.

---

## Goals

- Fit at production scale on machines without NVIDIA hardware (Apple
  silicon via MPS, plus CPU as an explicit opt-in).
- One numeric code path across cuda/mps/cpu so the CPU-device CI run
  certifies the exact code the GPU executes.
- Honor the full QEP-068 backend contract: constraint array layout,
  `BackendFitOutput` dtypes, states convention (0 converged), GHz units,
  float32 precision, `options.tolerance`/`max_number_iterations` semantics.
- Keep torch optional: a new `gpu` extra; core install stays light.
- `backend='auto'` gains a safe fallback: gpufit if available, else torch
  **only when a real GPU device (cuda/mps) exists**, else a clear error.

## Non-goals

- No analytic Jacobians (generic finite differences first; analytic can be a
  follow-up optimization).
- No MLE estimator in torch (LSE only; MLE warns and falls back, matching
  ScipyBackend).
- No bit-identical parity with gpufit — agreement is defined by the existing
  consistency-test contract (`rtol=1e-2`, `atol=1e-5`, `chi2 < 1e-6` on
  noiseless synthetic data).
- No removal or change of GpufitBackend/ScipyBackend.

---

## Design

### Packaging

```toml
[project.optional-dependencies]
gpu = ["torch>=2.4"]
```

`TorchBackend.is_available()` checks `importlib.util.find_spec("torch")`
without importing it (torch import costs 1-5 s; `qdmpy` import must never pay
it). `torch` is imported only inside method bodies — the same isolation
pattern `GpufitBackend` uses for pygpufit.

### Backend surface

```python
class TorchBackend:
    name = "torch"
    install_hint = "Install the GPU extra: uv sync --extra gpu (or pip install 'qdmpy[gpu]')."

    def __init__(self, device: str | None = None, chunk_size: int = 262_144): ...
```

- `device=None`/`'auto'` resolves cuda → mps → cpu at fit time; an explicit
  but unavailable device raises `DependencyError`.
- `supports(model)` returns `True` unconditionally. The first
  `model.func(x_t, p_t)` evaluation is guarded instead: an exception, or a
  return value that is not a torch tensor (a model that silently coerced to
  numpy/CPU), raises `DependencyError` explaining the framework-neutral
  contract and naming `backend='scipy'` as the fallback. A capability probe
  in `supports()` would add a torch import to a hot config path and could
  not distinguish "chokes on tensors" from any other model bug.

### Batched Levenberg-Marquardt

Float32 on every device (MPS has no float64; gpufit passes the consistency
contract in float32, so parity holds). Per chunk of `c` fits:

1. `p = clamp(p0, lo, hi)` — tolerate out-of-bounds initial guesses
   (gpufit/ScipyBackend parity). Bounds come from the shared
   `bounds_from_constraints()` mapping (extracted from ScipyBackend).
2. Jacobian by batched forward differences: `h_j = sqrt(eps_f32) * max(|p_j|,
   1e-6)`, `J[:, :, j] = (f(p + h_j e_j) - f0) / h_j` — n_params extra model
   evaluations per iteration, generic for any model.
3. Damped normal equations `(JᵀJ + λ·diag(JᵀJ)) Δ = -Jᵀr`, solved with
   `torch.linalg.cholesky_ex` + `cholesky_solve`. `cholesky_ex` returns a
   per-batch-element `info` code instead of raising, so a single singular
   pixel (dead pixel, zero contrast) cannot abort a 262k-fit batch; failed
   factorizations are treated as rejected steps (λ grows, restoring positive
   definiteness). Diagonal floored at 1e-20.
4. Box constraints by projection: `p_trial = clamp(p + Δ, lo, hi)`.
5. Marquardt schedule per fit: λ init 1e-3, x10 on rejected step, /10 on
   accepted, capped at 1e7.
6. Convergence per fit, on an accepted step:
   `|Δchi2| < tolerance * max(chi2, 1)` — the gpufit criterion, so
   `FitSettings.tolerance` keeps one meaning across backends.
7. Converged/invalid fits are frozen by masking (`torch.where`); tensor
   shapes stay static. States: 0 converged, 1 max-iterations, 2 non-finite
   input or model output.

**MPS caveat:** batched Cholesky is not implemented on MPS in all torch 2.x
versions. On the first failing factorization the backend caches
`_solve_on_cpu = True` and thereafter moves only the small `(c, p, p)` /
`(c, p)` solve tensors to CPU (~44 MB per iteration at c=262144, p=6); model
evaluations and the Jacobian — the expensive parts — stay on the GPU.

**Chunking:** fits are row-independent, so the backend loops over
`chunk_size` fits at a time (default 2^18 ≈ 0.7 GB peak at 50 freqs /
6 params). CUDA OOM errors are wrapped with a hint naming
`TorchBackend(chunk_size=...)`.

### Framework-neutral models

`models.py` gains `_ensure_2d(parameter)`: lists/tuples go through
`np.asarray` (preserving current behavior exactly), anything with `.ndim`
is reshaped without conversion — torch tensors pass through untouched,
whereas `np.atleast_2d` would silently copy them to CPU numpy. The three
built-in model functions switch to it, and the custom-model contract in the
`Model` docstring documents the rule: only framework-neutral arithmetic and
broadcasting.

### `'auto'` resolution

`resolve_backend(None | 'auto')` now returns an `AutoBackend` — a lazy
delegate that picks, at first availability check (never at construction,
preserving QEP-029's config/execution split):

1. `GpufitBackend` if pygpufit is importable;
2. else `TorchBackend(device='auto')` if torch is installed **and**
   `torch.cuda.is_available() or torch.backends.mps.is_available()`;
3. else unavailable — `DependencyError` listing all three fixes (install
   pyGpufit; install the `gpu` extra; or explicitly opt into CPU fitting
   with `backend='torch'` or `backend='scipy'`).

Torch-CPU is never auto-selected: a full frame on CPU can take minutes, and
QEP-068's "no silent slow fallback" rule stands. `FitSettings.backend` gains
the `'torch'` literal.

The workflow-level pygpufit preflight
(`measurement_workflows._backend_needs_gpufit_preflight`) narrows to apply
only when `backend='gpufit'` is explicit — with `'auto'` legitimately
resolving to torch, preflighting pygpufit for `'auto'` would wrongly block
torch-capable machines. Fail-fast is preserved by
`FitManager._require_backend_available()`, whose `DependencyError` now
appends the backend's `install_hint`.

---

## Alternatives Considered

1. **JAX** — elegant vmap/jit fit, but `jax-metal` (Apple GPU) is
   experimental and unreliable, and QEP-024 already rejected JAX on
   platform-support grounds. Fails the architecture-independence requirement
   in practice.
2. **CuPy** — smallest code delta from numpy, and QEP-024 proposed it for the
   guess stage; but it is NVIDIA-only (ROCm experimental, no Apple GPU),
   which is exactly the gpufit limitation this QEP exists to remove.
3. **Per-model torch kernels** — hand-written torch implementations of each
   ESR model (mirroring gpufit's model_id registry). Rejected: duplicates
   physics in two frameworks, and closes the open/closed door QEP-068 opened
   — custom models would again need backend-specific code.
4. **Auto-fallback to torch-CPU** — always works, but silently turns a
   seconds-long fit into minutes on GPU-less machines. Explicit opt-in only.
5. **Vectorized numpy/numba LM (no torch)** — fast on CPU and dependency-free,
   but no GPU; does not satisfy the requirement. The torch CPU device covers
   this niche anyway.

---

## GUI Integration Requirements

1. **Core API/data contract touchpoints:** `qdmpy-gui` calls
   `Measurement.fit_odmr`/`fit_folded_odmr`/`refit_outliers` and reads
   `QDMResult`. Result shapes, parameter names, states convention, and
   quality metrics are unchanged. Surface additions: `'torch'` as a valid
   `backend` value and `fit.backend` settings literal.
2. **Settings/migration:** `fit.backend='auto'` remains the default and now
   succeeds on Apple-silicon machines with the `gpu` extra installed instead
   of raising. No persisted-settings migration required. A GUI backend
   selector, if added, binds to the existing `fit.backend` key with the new
   `'torch'` choice.
3. **Error/progress behavior:** on machines with neither pygpufit nor a
   torch GPU device, fits raise a single `DependencyError` whose message
   lists the three remedies (pyGpufit, `gpu` extra, explicit CPU opt-in).
   Torch fits report the same progress granularity as gpufit (one backend
   call per frequency range); no new progress API.
4. **Acceptance checks:** on an Apple-silicon machine with the `gpu` extra:
   load → `fit_odmr()` (auto-resolves to torch/MPS) → inspect B111 maps →
   save `.qdm` → reload. Parameter maps agree with a gpufit reference within
   the consistency-test tolerances (`rtol=1e-2`, `atol=1e-5`); convergence
   rate on real data comparable to gpufit. On CUDA machines: no behavior
   change (gpufit still wins `'auto'`).
5. **Regression rationale:** the auto-semantics change only affects machines
   where fitting previously *failed*; machines where it worked keep gpufit.
   A GUI smoke test on a gpufit machine confirms no regression.

---

## Testing Plan

- **Unit (no torch installed):** `is_available()` False; `fit()` raises
  `DependencyError` naming the `gpu` extra; `resolve_backend('torch')`.
- **Unit (torch, CPU device):** LM recovers known Lorentzian parameters from
  perturbed guesses (`states==0`, consistency tolerances); constraint
  clamping honors constraint types (FREE columns ignored); out-of-bounds
  initial guesses clipped; **chunked ≡ unchunked** results; MLE→LSE warning;
  NaN spectra → state 2; output dtype/shape contract; device auto-selection
  matrix (mocked cuda/mps availability); custom pure-Python model fits
  end-to-end via `backend='torch'`; numpy-coercing model raises the
  framework-neutral error.
- **Consistency:** `tests/integration/test_torch_consistency.py`, sharing a
  parametrized `_run_consistency` helper with the gpufit variant — all three
  models, from true params and from perturbed guesses, on every locally
  available device (`cpu` always; `mps`/`cuda` when present).
- **Import hygiene:** subprocess test asserting `import qdmpy` does not
  import torch.
- **Auto-resolution matrix:** gpufit present → gpufit; absent + torch GPU →
  torch; neither → unavailable with the triple-remedy hint.

## Migration Plan

- Phase A: `_ensure_2d` in models.py (pure refactor, zero behavior change).
- Phase B: `torch_backend.py` + `gpu` extra + unit/consistency tests.
- Phase C: registry/AutoBackend/settings/preflight rewiring + test updates.
- Each phase gated on `uv run pytest` green.

## Success Criteria

- A full synthetic frame fits on Apple-silicon MPS in seconds-to-tens-of-
  seconds (vs. hours on ScipyBackend), with convergence rate and parameter
  agreement matching the consistency contract. ✅ (see measured numbers below)
- `import qdmpy` cost unchanged (torch lazily imported). ✅ (subprocess test)
- `backend='auto'` works on this repo's development Mac without pygpufit. ✅
- All existing tests green; gpufit path byte-for-byte untouched. ✅

## Implementation Notes (2026-07-04)

Measured on the development Mac (Apple silicon, torch 2.12.1): 256x256 x
2-pol x 2-range synthetic ESR14N frames fit at **~31,000 fits/s on MPS**
(~5 min extrapolated for a full 1200x1920 frame) and ~21,500 fits/s on
torch-CPU, both at 100% convergence; B111 maps agree with ScipyBackend
within 0.1 µT on a 16 µT-std field. The LM engine went through several
empirically-driven revisions worth recording:

1. **Non-strict step acceptance (`chi2_trial <= chi2`).** At the float32
   chi2 floor, trial steps produce bit-identical chi2; with strict `<` those
   fits reject forever and the accepted-step convergence check never fires
   (observed: 63% of fits burning 1000 iterations with perfect parameters).
2. **Three-route convergence criterion.** (a) bitwise-equal chi2 on an
   accepted step (float32 floor — lambda-independent); (b) small-but-nonzero
   delta-chi2 only in the Gauss-Newton regime (`lambda <= lambda_init`) —
   without this guard, micro-steps at large lambda right after a rejection
   streak satisfied delta-chi2 < tol *far from the optimum* (observed:
   fits stopping at chi2 ~7e-4 after 5 iterations); (c) a 10-rejection
   stall terminator.
3. **Active-set reduction for box constraints.** A parameter pinned at a
   bound with an outward gradient makes every projected trial step uphill
   (reject forever). Its Jacobian column is zeroed for that iteration so
   the free parameters keep optimizing. Discovered via the default mT-mode
   width window, which pins `width` for 100% of pixels on synthetic data.
4. **Absolute finite-difference step floor (1e-5).** A relative-only step
   dies for near-zero parameters: for offset ~1e-4 the perturbed model value
   is bitwise-unchanged in float32, the Jacobian column is exactly zero, and
   the parameter can never move (observed: offsets frozen at their initial
   values).
5. **Working-set compaction every 8 iterations, not masking.** With pure
   masking, a 600-iteration convergence tail forces the entire 262k-fit
   chunk through every iteration (a 256x256 frame took >10 min). Host-side
   checks sync the GPU pipeline, so compaction happens at window boundaries
   (one sync per 8 iterations), with masked freezing in between.
6. **MPS routes the Cholesky solve to CPU.** `torch.linalg.cholesky_ex` on
   MPS is ~80x slower than a CPU round-trip for many-small-matrix batches
   (measured 234 ms vs 3 ms at 8192x6x6); only the (c, p, p) solve moves,
   model evaluations and Jacobians stay on the GPU. The try/except fallback
   remains for other devices with missing linalg support.
7. **`lambda_down = 3` instead of the classic 10** — benchmarked 2.8x fewer
   iterations at identical final chi2 (aggressive relaxation overshoots and
   oscillates accept/reject near the optimum).
8. The stacked-Jacobian trick (all n_params perturbations in one model
   call) cuts kernel dispatches per iteration ~6x, which matters on
   dispatch-bound MPS.
