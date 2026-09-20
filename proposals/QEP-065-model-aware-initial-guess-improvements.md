# QEP-065 -- Model-Aware Initial Guess Improvements (Contrast First, Width Next)

| Field   | Value |
|---------|-------|
| Status  | Draft |
| Created | 2026-03-11 |
| Scope   | `qdmpy.fitting.guesser`, `qdmpy.fitting.guess`, fit benchmarking scripts |
| Depends | QEP-024 |

## Motivation

ODMR fits show sensitivity to initial parameters, especially on large scans where
local failures become spatial artifacts. Current initialization is robust and
fast, but multi-peak models still use a coarse contrast seed (same total
contrast copied to each peak), and width estimation remains a likely bottleneck
for hard spectra.

The immediate goal is to improve guess quality for `ESR14N`, `ESR15N`, and
`ESRSINGLE` without breaking runtime expectations for large images
(up to ~1200x1920 pixels).

## Current State

Initial guesses are generated in `ParameterGuesser` using:

- `center`: absorption centroid
- `contrast`: `top3_contrast` total dip depth
- `width`: envelope HWHM (`halfpower_width`) then AHYP correction for multi-peak
- `offset`: edge-baseline estimate

For multi-peak models, total contrast is currently copied into each
`contrast_i` parameter.

## Prototype Evidence (Per-Peak Contrast)

A prototype script was added:

- `scripts/prototype_per_peak_contrast.py`

Prototype method:

1. Keep baseline `center`, `width`, `offset` guesses.
2. Sample spectrum depth at expected dip positions (`center + shift_i`).
3. Solve overlap-corrected linear system for per-peak contrasts.
4. Clamp and fallback to baseline total-contrast behavior on instability.

Observed behavior:

- Synthetic truth-recovery improves strongly:
  - `ESR14N`: ~5.6x lower contrast MAE
  - `ESR15N`: ~3.5x lower contrast MAE
- Real fixtures (test datasets):
  - small but consistent final chi2 improvement,
  - no convergence-state change on current fixtures,
  - substantial iteration-count differences (optimizer trajectory changes),
  - runtime overhead in guess stage is measurable but acceptable.

## Problem Statement

We need a production-ready initial-guess strategy that:

1. Improves fit robustness on difficult spectra (not just easy fixtures).
2. Preserves or improves chi2 and convergence statistics.
3. Scales to large FOVs without unacceptable runtime or memory overhead.
4. Keeps behavior deterministic and testable.

## Proposal

Implement model-aware initial-guess improvements in phases, with explicit
benchmark gates between phases.

### Phase 1 -- Per-Peak Contrast Initialization (Primary)

- Add configurable contrast guess strategy in `ParameterGuesser`:
  - `total` (current behavior, default until validated)
  - `per_peak_sampled` (new strategy)
- Apply to:
  - `ESR14N`: estimate `contrast_0/1/2`
  - `ESR15N`: estimate `contrast_0/1`
  - `ESRSINGLE`: keep current single-contrast path
- Keep stable fallback to `total` for any numerical edge case.

### Phase 2 -- Width Guess Improvements (Priority Follow-Up)

Investigate width strategies that remain fast:

- local half-depth width around expected dip positions,
- model-aware envelope decomposition,
- optional two-candidate width selection using cheap proxy residual.

Constraints:

- no expensive per-pixel peak finding loops in production path,
- no major API break,
- preserve GHz internal units and model conventions.

### Phase 3 -- Combined Robustness Controls

- Add optional low-cost second-pass mechanisms for hard pixels:
  - targeted re-initialization for suspect pixels before full refit,
  - integration with existing outlier refit flow,
  - metrics hooks for failure-tail analysis (95p/99p chi2, non-converged count).

## Implementation Plan

1. Land Phase 1 as opt-in strategy with tests and benchmarks.
2. Run fixture matrix (14N + 15N + large FOV) and compare:
   - chi2 mean/median/tails,
   - convergence rate,
   - iteration counts,
   - guess-stage runtime.
3. Promote Phase 1 strategy to default only if acceptance criteria pass.
4. Design and prototype Phase 2 width strategies with the same benchmark gates.

## Acceptance Criteria

Phase 1 acceptance:

- On hard-case datasets, reduced non-converged pixel fraction and/or improved
  high-chi2 tail metrics versus baseline.
- No meaningful regression on existing 14N/15N regression fixtures.
- Guess-stage overhead remains bounded and acceptable for large FOVs.

Phase 2 acceptance (width):

- Additional robustness gain beyond Phase 1 alone.
- Runtime increase justified by improved failure-tail behavior.

## Testing Strategy

Data tiers:

1. Synthetic spectra with known per-peak contrasts and widths.
2. Existing real regression fixtures:
   - 14N: `tests/data/real_fov1_*`
   - 15N: `tests/data/real_fov18x_*`, `tests/data/FOV18x`
3. Large FOV runtime fixture:
   - `tests/data/MIL2_FOV1` (~1200x1920)

Metrics:

- `mean_chi2`, `median_chi2`, p95/p99 chi2,
- convergence rate (`states == 0`),
- iteration-count distribution,
- guess-stage runtime and total fit runtime.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Better per-peak contrasts do not improve final fit tails | Include tail metrics and hard-case datasets in acceptance gates |
| Runtime overhead too high on large FOVs | Keep strategy vectorized; enforce runtime budget checks on MIL2_FOV1 |
| Width remains dominant failure source | Prioritize Phase 2 immediately after Phase 1 evaluation |
| Numerical instability in overlap solve | Use clamping and deterministic fallback to `total` strategy |

## GUI Integration Requirements

- No immediate GUI changes required if new strategy is internal or defaulted.
- If strategy selection is exposed later, GUI should surface an advanced fit
  option with clear labels (for example `contrast_guess_strategy`).
- Result schema, map units, and coordinate conventions remain unchanged.
- GUI should continue to receive standard fit progress/error behavior.

## Out of Scope

- Full model reparameterization (for example independent per-peak widths in
  production kernels).
- Changes to B111 physics extraction conventions.
- Non-fitting pipeline changes unrelated to initialization.
