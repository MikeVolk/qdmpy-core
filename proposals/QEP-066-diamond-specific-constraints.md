# QEP-066 -- Diamond-Specific Fit Constraint Profiles

| Field   | Value |
|---------|-------|
| Status  | Draft |
| Created | 2026-03-12 |
| Scope   | `qdmpy.settings`, `qdmpy.fitting.manager`, measurement-level fit configuration |
| Depends | QEP-065 |
| Related | QEP-062 |

## Motivation

A single global constraint set is suboptimal across diamond/isotope conditions.
Empirical analysis on full datasets (`FOV1`/14N and `FOV18x`/15N) shows
materially different fitted width and center distributions. Overly loose
constraints reduce robustness; overly tight shared constraints clip valid
solutions (especially for 15N width lower tails).

We need first-class, explicit diamond-specific constraint profiles with safe
fallback behavior.

## Problem Statement

Current constraints are global and model-agnostic. Users cannot cleanly select
tuned constraints for 14N vs 15N (or future diamond classes) without manual
parameter editing.

This causes:

- avoidable fit sensitivity to initialization,
- inconsistent behavior across datasets,
- friction for reproducible pipelines and GUI presets.

## Goals

1. Add configurable, versioned constraint profiles keyed by diamond/isotope class.
2. Preserve backward compatibility for existing explicit constraint dicts.
3. Support automatic profile selection where model is known/resolved.
4. Keep runtime overhead negligible.
5. Provide clear diagnostics showing which profile was applied.

## Non-Goals

- Reparameterizing model kernels.
- Replacing user explicit constraints.
- Introducing GUI-only logic in core.

## Proposed Design

### 1) New profile concept

Introduce a profile layer above raw constraints:

- `constraint_profile`: string key (for example `global_default`, `n14_default`,
  `n15_default`)
- `constraint_profile_mode`: `auto | explicit`
- `constraint_profile_overrides`: optional per-parameter patch applied after
  profile load

Resolution order:

1. User `constraints` argument (highest priority, current behavior preserved)
2. Profile + overrides
3. Existing global settings fallback

### 2) Built-in profiles (initial)

Initial profile values (from current benchmarking):

- `global_default_tight` (single *N-safe):
  - `center_min_mt=0.0`
  - `center_max_mt=1.10`
  - `width_min_mt=0.017`
  - `width_max_mt=0.080`
  - `contrast_min=0.003`
- `n14_default` (seeded from FOV1 family)
- `n15_default` (seeded from FOV18x family)

`auto` mapping (initial):

- model `ESR14N` -> `n14_default`
- model `ESR15N` -> `n15_default`
- model `ESRSINGLE` -> `global_default_tight` (until dedicated profile exists)

### 3) API surface

At minimum:

- Settings schema additions under model/constraints profile section.
- Optional `fit_odmr(..., constraint_profile='...')` convenience argument.
- Logging in `FitManager`:
  - selected profile key,
  - effective resolved bounds after unit conversion.

### 4) Validation and safety

- Validate profile names at startup/fit call.
- Fail with clear error on unknown profile.
- Guarantee deterministic merge order (profile -> overrides -> explicit
  per-call constraints).

## Implementation Plan

1. Add typed profile schema and built-in registry in settings layer.
2. Add profile resolution utility returning normalized constraints.
3. Wire profile resolution into `FitManager` initialization.
4. Add logging of effective constraints.
5. Add tests for:
   - auto profile selection by model,
   - explicit profile selection,
   - override precedence,
   - backward compatibility with current `constraints` dict.
6. Add regression benchmark script mode:
   - compare profiles on FOV1/FOV18x,
   - report chi2 mean/p95/p99, convergence, boundary-hit fractions, runtime.

## Acceptance Criteria

1. Functional:
   - Profiles selectable and applied deterministically.
   - Existing calls without profiles behave as before.
2. Quality:
   - `n14_default` and `n15_default` show non-regression on respective fixtures.
   - `global_default_tight` remains safe for mixed usage.
3. Observability:
   - Fit logs show chosen profile and effective numeric bounds.
4. Performance:
   - No measurable overhead beyond config resolution.

## Testing Strategy

Fixtures:

- 14N: `tests/data/real_fov1_*`, `tests/data/MIL2_FOV1`
- 15N: `tests/data/real_fov18x_*`, `tests/data/FOV18x`

Metrics:

- convergence rate,
- non-converged count,
- chi2 mean/median/p95/p99,
- width/center boundary-hit fractions,
- total fit time.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Wrong auto mapping for uncommon data | Keep explicit profile override and explicit constraints highest priority |
| Profile drift over time | Version profile definitions and track benchmark results |
| User confusion from multiple configuration paths | Document precedence and log resolved constraints clearly |
| Over-tight profile clipping valid physics | Include boundary-hit metrics in acceptance gates |

## GUI Integration Requirements

- GUI should expose optional advanced selector for `constraint_profile`.
- GUI must continue supporting manual per-parameter overrides.
- Display active profile in fit metadata/status for reproducibility.
- No changes to output map schema or units.

## Out of Scope

- Automatic diamond-type classification from raw data.
- Per-pixel adaptive constraints.
- New model physics.
