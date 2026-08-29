# QEP-073 - Analytic Jacobians for the ESR Models

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-068 (fit backend seam), QEP-069 (torch fit backend) |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-08-29 |
| **Implemented** | 2026-08-29 |

---

## Motivation

QEP-069 shipped `TorchBackend` with generic forward finite-difference Jacobians, listing
analytic Jacobians as an explicit non-goal ("generic finite differences first; analytic can
be a follow-up optimization"). A same-machine benchmark of `GpufitBackend` against
`TorchBackend` (RTX 3050 Ti, LSE pinned on both, perturbed initial guesses, 100k fits)
shows that decision is now costing accuracy, not just speed:

| model | gpufit chi2 max | torch chi2 max | gpufit max center err | torch max center err |
|---|---|---|---|---|
| ESRSINGLE | 8.1e-13 | 3.6e-12 | 0 | 0 |
| ESR14N | 1.06e-4 | 1.43e-4 | 1.19e-3 | 1.56e-3 |
| ESR15N | 2.48e-4 | **1.31e-3** | 1.23e-3 | **2.45e-3** |

ESR15N is the *simpler* model -- 2 dips and 5 parameters against ESR14N's 3 dips and 6 -- so
`TorchBackend` being worse on it points at an implementation defect rather than intrinsic
difficulty.

### Root cause: the finite-difference step is scaled to the wrong quantity

`torch_backend.py:315` computes the step relative to each parameter's own magnitude:

```python
step = torch.clamp(_FD_EPS * p_w.abs(), min=_FD_MIN_STEP)  # _FD_EPS = 3.45e-4
```

For `center` this is the problem. Centers are ~2.87 GHz, so:

```
step = 3.45e-4 * 2.87 = 9.9e-4 GHz ~ 0.99 MHz
```

The comment at `torch_backend.py:42-48` justifies the `1e-5` *floor* as "far below every
physical parameter scale (centers ~2.87 GHz, widths ~2e-3 GHz ...)". That reasoning is sound
for the floor but does not apply to the relative term: `center`'s relevant scale is not its
magnitude (2.87) but the **linewidth** (~6e-4 to 1.2e-3 GHz). The Jacobian column for the
single most important parameter is differenced across roughly one linewidth.

Real fitted widths make this concrete. `reference_data/FOV18x_reference_bin2.npz` gives a
median 15N HWHM of **0.615 MHz**; QEP-011 (lines 633-634) quotes ~0.6 MHz for 15N and ~1.2
MHz for 14N. So on real data the `center` FD step of 0.99 MHz is **1.6x the 15N HWHM**.

### Why ESR15N specifically

The hyperfine geometry, from `constants.py:22-23` and confirmed against the gpufit kernels
(`Gpufit/models/esr14N.cuh:109`, `esr15N.cuh:109`):

- **ESR14N** (`ahyp = 0.002158` GHz): dips at `center-ahyp`, **`center`**, `center+ahyp`.
  Adjacent spacing 2.158 MHz, envelope 4.316 MHz.
- **ESR15N** (`ahyp = 0.0015` GHz): dips at `center-ahyp`, `center+ahyp`. Separation 3.000
  MHz. **No dip at `center`.**

Note the 15N doublet is *more* resolved than 14N's adjacent spacing, so resolution is not
the explanation. The explanation is that 14N has a dip sitting exactly at `center`, making
the center directly observable, whereas 15N must infer `center` from the symmetry of two
flanking dips. That inference is far more sensitive to derivative error.

### Secondary benefit: speed

Forward differences cost `n_params + 1` model evaluations per LM iteration -- 7x for ESR14N.
`TorchBackend` also needs more iterations than gpufit (8.5 vs 5.5 mean on ESR14N), plausibly
because the inaccurate `center` column degrades step quality. A same-machine benchmark put
the pure-fitting gap at 3.6x at full-frame scale (4.6M fits: 7.6 s gpufit vs 27.3 s torch).
Analytic Jacobians address both.

Note this is *not* urgent on throughput grounds: end-to-end on real full-resolution data the
two backends are within 1.11x (121 s vs 134 s), because loading, guessing and B111 dominate.
Accuracy is the reason to do this.

---

## Goals

- Remove finite-difference error from `center` (and every other parameter) for the three
  built-in ESR models.
- Preserve the framework-neutral `Model` contract: the same code path must evaluate numpy
  arrays and torch tensors, with no framework imports in `models.py`.
- Keep finite differences working for custom models -- the documented custom-model contract
  must not regress.
- Reduce the fitting-time gap to `GpufitBackend` as a side effect.

## Non-goals

- No change to fit results' physics conventions, units (GHz throughout), or the
  `BackendFitOutput` contract.
- No change to `GpufitBackend`; gpufit's kernels remain the numerical reference.
- No second-derivative / exact-Hessian work -- Gauss-Newton `JᵀJ` stays as is.
- No new models. QEP-072 (dip-peak windowed fitting) is independent.

---

## Design

### One shared Jacobian, not three

All three ESR models are the same functional form: a sum of Lorentzian dips at fixed offsets
from `center`, sharing one HWHM `width` and one `offset`.

```
f(x) = 1 + offset - sum_i c_i * w^2 / ((x - center - delta_i)^2 + w^2)
```

with `delta_i` in `{-a, 0, +a}` (ESR14N), `{-a, +a}` (ESR15N), `{0}` (ESRSINGLE). So the
implementation is **one** helper parameterized by the dip offsets, not three transcribed
kernels.

Writing `d_i = x - center - delta_i` and `D_i = d_i^2 + w^2`:

```
df/dcenter = -sum_i c_i * 2 w^2 d_i / D_i^2
df/dw      = -sum_i c_i * 2 w d_i^2 / D_i^2
df/dc_i    = -w^2 / D_i
df/doffset = 1
```

These were derived from the model form and then checked term by term against gpufit's
analytic derivatives at `Gpufit/models/esr14N.cuh:130-136`. They agree; gpufit writes
`df/dw` in the algebraically equivalent form `2 w^3 c_i / D_i^2 - 2 w c_i / D_i`. **gpufit is
the reference implementation** -- if the new tests disagree, our derivation is wrong.

### Framework neutrality: return columns, let the backend stack

`Model.func` already honors a "framework-neutral arithmetic only" contract, enforced at
`torch_backend.py:240-268`, so one implementation serves numpy and torch. A Jacobian cannot
honor that contract while calling `np.stack` or `torch.stack`.

So `Model.jacobian` returns a **tuple of per-parameter derivative arrays**, one `(n_fits,
n_x)` array per entry of `parameter_names`, in that order. Each backend stacks with its own
framework: `torch.stack(cols, dim=-1)`, `np.stack(cols, axis=-1)`. The helper itself uses
only arithmetic and broadcasting.

### An optional hook, with FD as the fallback

```python
class Model(ABC):
    def jacobian(self, x, parameters) -> tuple | None:
        """Per-parameter analytic derivatives, or None to use finite differences."""
        return None
```

Custom models that do not implement it keep working unchanged. `TorchBackend` resolves the
hook **once per `fit()`** (not per LM iteration) and falls back to `_fd_jacobian` when it
returns `None` or raises.

`_fd_jacobian` is kept, not deleted: it is the custom-model fallback and the reference the
new tests check the analytic form against.

### Where it plugs in

There is exactly one Jacobian call site in the LM loop, `torch_backend.py:388`. Downstream
code consumes the existing `(a, f, p)` layout (`jac.mT @ jac`, active-set column zeroing at
`:398`), so the analytic path must produce that same layout and nothing else changes.

`ScipyBackend` gains `jac=` on its `least_squares` call (`backends.py:253`) when the model
supplies one -- nearly free, and keeps the CPU path consistent.

### Memory layout is part of the contract, not an implementation detail

The obvious way to assemble the columns, `torch.stack(cols, dim=-1)`, gives the right shape
(a, f, p) and is **2.6x slower end-to-end** than finite differences. `_fd_jacobian` returns
`(...).permute(1, 2, 0)`, which is (a, f, p) in shape but (p, a, f) in memory; the downstream
`jac.mT @ jac` inherits that layout and cuBLAS gets contiguous rows. Stacking straight to the
last dim leaves each fit's matrix column-major for the batched matmul, and profiling shows
`aten::bmm` going from 79 ms to 394 ms -- 74% of all device time, swamping everything the
analytic form saves.

So `_analytic_jacobian` stacks on dim 0 and permutes, matching `_fd_jacobian` exactly. Any
future change to either must preserve that layout.

---

## Risks

- **A wrong derivative is worse than a coarse one**: it converges confidently to the wrong
  answer. Mitigated by testing every column of every model against float64 central
  differences, and by the existing gpufit consistency suite.
- **Degenerate regimes**: at `w -> 0`, `D_i -> d_i^2` and derivatives blow up near `x =
  center + delta_i`. The existing `width` lower constraint (1e-4 GHz) keeps us away from
  this, and the LM damping term handles conditioning. Verify no new `SINGULAR_HESSIAN` /
  non-converged states appear.
- **Benchmark regime was unrealistic**: the existing consistency-test width range
  (0.002-0.005 GHz HWHM) is 3-8x wider than real data (~0.0006 GHz for 15N). Conclusions
  drawn only at those widths are suspect in both directions, so the benchmark gains a
  realistic-width case rather than replacing the old one.

---

## Implementation steps

1. `_lorentzian_dips_jacobian(x, parameters, dip_offsets)` plus `esr14n_jacobian`,
   `esr15n_jacobian`, `esrsingle_jacobian` in `src/qdmpy/fitting/models.py`, beside the
   existing `esr14n` / `esr15n` / `esrsingle`.
2. `Model.jacobian()` on the ABC returning `None`; overrides on `ESR14N`, `ESR15N`,
   `ESRSINGLE`.
3. Tests first: analytic vs float64 central differences, per model, per parameter.
4. Wire the analytic path into `TorchBackend` at the `:388` call site, FD retained as
   fallback.
5. Wire `jac=` into `ScipyBackend`.
6. Add `--width-range` to `scripts/benchmark_fit_backends.py`; measure both regimes,
   before and after.
7. Tighten `tests/integration/test_torch_consistency.py` tolerances to lock the gain in.
8. CHANGELOG entry under `## [Unreleased]`.

## Results

Measured on the same machine (RTX 3050 Ti), LSE, perturbed start, 100k fits, synthetic
widths -- the regime the "before" table above was taken in.

| model | metric | before | after | gpufit |
|---|---|---|---|---|
| ESR14N | chi2 max | 1.43e-4 | **1.06e-4** | 1.06e-4 |
| ESR14N | max center err (GHz) | 1.56e-3 | **1.18e-3** | 1.19e-3 |
| ESR14N | mean LM iterations | 8.5 | **6.5** | 5.5 |
| ESR15N | chi2 max | 1.31e-3 | **2.48e-4** | 2.48e-4 |
| ESR15N | max center err (GHz) | 2.45e-3 | **1.23e-3** | 1.23e-3 |
| ESRSINGLE | chi2 max | 3.55e-12 | **7.53e-13** | 8.06e-13 |

TorchBackend now matches gpufit to the reported precision on every model. Fitting is 1.86x
(ESR14N) / 1.75x (ESR15N) / 1.03x (ESRSINGLE) faster, and the LM iteration tail collapses:
ESR14N max 245 -> 63, p99.9 142 -> 25.

At realistic widths the gain is larger still, as predicted -- ESR14N chi2 max 2.58e-2 ->
2.12e-3, ESR15N 7.30e-3 -> 5.55e-4, and the number of fits with a center error above 1e-3
GHz falls from 1662 to 253 (ESR14N). The residual worst-case errors that remain in that
regime are a sampling artifact, not a derivative one: 50 points across 40 MHz is 0.8 MHz
spacing against a 0.6 MHz HWHM, so a handful of pixels out of 100k sit in local minima on a
barely-sampled line. gpufit's float64 Hessian accumulators handle those few better; the
analytic path still beats finite differences there (ESRSINGLE max center error 6.85e-2 ->
2.48e-2).

Real data, gpufit vs torch, bin_factor=2:

| dataset | metric | before | after |
|---|---|---|---|
| MIL2_FOV1 (14N) | B111 remanent / induced r | 0.993 / 0.997 | **0.9997 / 0.9999** |
| FOV18x (15N) | B111 remanent / induced r | 0.52 / 0.80 | **0.974 / 0.991** |
| FOV18x (15N) | remanent map spread | 2x gpufit's | **0.1034 vs 0.1057 uT** |

FOV18x's median per-pixel center difference between backends is 0.00024 MHz. Note the
correlation caveat still stands -- that map's spread is near the fit noise floor -- which is
why the spread agreement is reported alongside `r`.

### End-to-end, and an estimator asymmetry found while measuring it

MIL2_FOV1 at `bin_factor=4` (576k spectra), measurement loaded once and reused:

| run | backend fit time | chi2 median | converged |
|---|---|---|---|
| gpufit, MLE (the default) | 2.18 s | 5.04e-7 | 100% |
| torch, LSE (its fallback) | 3.10 s | 9.99e-7 | 100% |
| gpufit, LSE pinned | 4.28 s | 9.9930e-7 | 100% |
| **torch, LSE pinned** | **2.91 s** | 9.9921e-7 | 100% |

At the same estimator `TorchBackend` is now 1.47x *faster* than `GpufitBackend` on real
data, at equal chi2. Two things this exposes:

1. `QDMpySettings.fit.estimator` defaults to `"MLE"`, which `TorchBackend` does not
   implement -- it warns and runs LSE. So the shipped default compares gpufit's MLE against
   torch's LSE, which is not a like-for-like comparison and is also why torch's chi2 looked
   worse before the estimator was pinned. The backends are not interchangeable on the
   default configuration; MLE is the correct estimator for photon-shot-noise-limited data.
   This predates QEP-073 and warrants its own follow-up.
2. gpufit's own LSE path is ~2x slower than its MLE path (4.28 s vs 2.18 s), so "gpufit is
   faster by default" is really "gpufit's MLE kernel is fast", not a general kernel edge.

Keep the throughput in proportion: loading this measurement takes 16 s against 3-4 s of
fitting, so at bin 4 the backend choice moves end-to-end wall time by roughly 5%. Accuracy
was the reason for this work; the speed is a side effect.

## Acceptance

- ESR15N torch chi2 max falls from 1.31e-3 to gpufit's order (~2.5e-4) or better.
- ESR15N torch max center error falls from 2.45e-3 toward gpufit's 1.23e-3; ESR14N from
  1.56e-3 toward 1.19e-3.
- Mean LM iterations fall from 8.5 (ESR14N) toward gpufit's 5.5.
- Throughput improves; any regression is a bug.
- Convergence stays at 100% wherever it already was.
- On real data: MIL2_FOV1 (14N) B111 agreement with gpufit does not regress from r = 0.993
  remanent / 0.997 induced; FOV18x (15N) improves. Judge FOV18x primarily on the fitted
  `center` parameter and map spread -- its B111 spread (0.157 uT) sits near the fit noise
  floor, so correlation on a near-flat map is a weak metric.
