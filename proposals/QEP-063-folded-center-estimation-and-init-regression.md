# QEP-063 -- Folded Center Estimation and Initialization Regression Investigation

| Field   | Value |
|---------|-------|
| Status  | Draft |
| Created | 2026-03-10 |
| Scope   | `qdmpy.odmr.folding`, `qdmpy.fitting.manager.fit_folded`, folded real-data regressions |
| Depends | QEP-059, QEP-060 |

## Motivation

Folded fitting quality regressed on real 15N data (`FOV18x`) after recent folded
path and constraint changes. This QEP captures what is confirmed, what was fixed,
and what remains unresolved, so work can restart without re-discovery.

Primary symptom:

- folded-vs-normal B111 agreement is poor on 15N fixtures,
- while 14N fixtures remain mostly good.

## Confirmed Facts (Current State)

### 1) Constraint propagation bug in folded fit was real and is fixed

`fit_folded()` previously rebuilt a `FitManager` that dropped active user
constraints (for example explicit `center` bounds). This was fixed by preserving
active constraints and only overriding folded-specific `contrast/offset` bounds.

Files:

- `src/qdmpy/fitting/manager.py`
- `tests/test_folded_fit.py` (regression added)

### 2) GHz conversion at fit boundary is correct

Folded fit call uses absolute GHz axis (`D_ZFS + delta_f`) and folded initial
center guesses are in GHz. No Hz/GHz mismatch was observed at gpufit boundary.

### 3) Behavior differs strongly by dataset/isotope

- `FOV1` (14N): folded-vs-normal is generally good.
- `FOV18x` (15N): folded-vs-normal is consistently poor with current brute-force
  fold-center map path.

## Real-Data Fixtures (for fast restart)

All fixtures are 128x128 crops and loader-verified against their source crops.

14N fixtures:

- `tests/data/real_fov1_fov2037485_x365y1061`
- `tests/data/real_fov1_fov1069295_x1775y556`
- `tests/data/real_fov1_fov1074295_x1015y559`

15N fixtures:

- `tests/data/real_fov18x_fov5838_x78y24`
- `tests/data/real_fov18x_fov7539_x99y31`
- `tests/data/real_fov18x_fov14925_x45y62`

Regression test file:

- `tests/integration/test_folded_real_data_regression.py`

## Prototype Findings (scripts/prototype_fft_v2.py)

### 15N (`FOV18x`) summary

- Current brute-force fold center is worst (high induced RMSE, low corr).
- `FFT+refine` improves quality strongly but is slow (~4.3-4.5 s per 128x128 fixture).
- Updated `FFT-zp-8x` is much better than brute-force and far faster than `FFT+refine`.
- Centroid family performed best in quality/speed tradeoff on tested 15N fixtures:
  - `centroid-p2` best induced RMSE/corr among fast methods,
  - runtime is very low (~40-50 ms for D-map estimation).

### 14N (`FOV1`) summary

- Baseline brute-force already good.
- Plain `centroid` is robust and competitive.
- `centroid-p2` can fail on at least one fixture (outlier behavior).

## Folding-Center vs Fit-Initialization Clarification

Two separate decisions must not be conflated:

1. **How to estimate fold center `D(y,x)`** (folding stage)
2. **How to initialize folded fit center parameter** (fit stage)

Important constraint:

- Do **not** use raw `d_zfs_map` directly as folded fit `center` initial parameter.
- This produced catastrophic errors in tests because folded fit `center`
  parameterization is not equal to raw local `D`.

## Problem Statement to Solve

Need a production folded-center strategy that:

- materially improves 15N (`FOV18x`) folded-vs-normal agreement,
- does not regress 14N (`FOV1`) behavior,
- preserves acceptable runtime for large scans.

## Proposed Direction (Next Iteration)

### A) Add explicit folded center estimator strategy

Introduce setting (draft naming):

- `fold_center_estimator = 'bruteforce' | 'centroid' | 'centroid_p2' | 'fft_zp8' | 'fft_refine' | 'auto'`

Draft `auto` policy:

- `ESR15N -> centroid_p2`
- `ESR14N -> centroid`

Use `model_name` to drive policy; no separate "diamond detector" required.

### B) Rework folded fit initial center guessing separately

Implement folded-fit center initialization compatible with folded fit
parameterization (not raw `d_zfs_map` assignment).

### C) Keep fixture-based regression guardrails

Use existing real-data fixtures as acceptance checks for both quality and runtime.

## Acceptance Criteria (Draft)

On the three 15N fixtures:

- reduce induced RMSE substantially vs current brute-force folded path,
- increase induced corr substantially vs current brute-force folded path,
- keep runtime below `FFT+refine` baseline by a wide margin.

On the three 14N fixtures:

- no statistically meaningful degradation vs current baseline.

On API/behavior:

- folded path remains deterministic under fixed settings,
- constraints remain honored in folded fit.

## Reproduction Commands

Primary benchmark:

```bash
uv run python scripts/prototype_fft_v2.py
```

Folded real-data regression test:

```bash
uv run pytest tests/integration/test_folded_real_data_regression.py -q --no-cov
```

## Risks

| Risk | Mitigation |
|------|------------|
| 15N-tuned strategy regresses 14N | Keep separate 14N fixtures in regression suite and gate on both families |
| Fast estimator introduces local artifacts | Compare map-level metrics and fold residual, not single scalar only |
| Incorrect folded init coupling | Keep folding-center estimation and folded-fit initialization as separate components |

## GUI Integration Requirements

- No GUI API break expected if default behavior remains automatic.
- If estimator becomes user-configurable, GUI may expose optional advanced control
  (follow-up, not required for core acceptance).
- Core must continue returning folded outputs and fitted B111 maps in current units
  and coordinate conventions.

## Out of Scope

- Rewriting full folded model parameterization.
- Changing normal (non-folded) fit behavior.
- New file formats or result schema changes.
