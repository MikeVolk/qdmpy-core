# QEP-064 -- Folded Range Symmetrization for Robust Fold-Center Estimation

| Field   | Value |
|---------|-------|
| Status  | Investigated (On Hold) |
| Created | 2026-03-11 |
| Scope   | `qdmpy.odmr.folding`, folded prototype benchmarking |
| Depends | QEP-059, QEP-060, QEP-063 |

## Motivation

Real 15N data (`FOV18x`) shows asymmetric low/high measurement ranges around
nominal `D_ZFS = 2.870 GHz`. This can bias fold-center estimation and degrade
folded-vs-normal fit agreement.

Confirmed example (`tests/data/FOV18x`):

- low range: `2.837..2.850` GHz
- high range: `2.887..2.900` GHz
- `D-low_max = 20.0 MHz`, `high_min-D = 17.0 MHz`
- asymmetry around `D_ZFS`: `~3.0 MHz`

By contrast, `FOV1` is close to symmetric (`~0.7 MHz` asymmetry).

## Confirmed Prior Findings

This QEP builds on QEP-063 investigation results:

- folded constraint propagation bug in `fit_folded()` was real and has been fixed,
- GHz conversion at folded fit boundary is correct,
- `FOV18x` (15N) is still poor with current brute-force folded center map,
- centroid-based folded center prototypes are promising,
- folded center estimation and folded-fit center initialization are separate concerns,
- raw `d_zfs_map` must not be used directly as folded fit `center` init.

## Problem Statement

Need a folded pre-processing strategy that compensates branch-range asymmetry,
improves 15N behavior, and avoids regressions on 14N.

## Prototype Findings (2026-03-11)

We evaluated a prototype matrix on 15N real-data fixtures (`FOV18x`) with:

- `fit_range`: native vs symmetric
- `fold_range`: native vs symmetric
- `D_ref`: mean-spectrum global center
- model: `ESR15N`

### Key result

Symmetric range restriction did **not** improve folded-vs-normal fit accuracy in
this prototype.

Average induced-field behavior over the three 15N fixtures:

- `fold native / fit native`: best (`RMSE ~0.86 uT`, `corr ~0.83`)
- `fold native / fit symmetric`: worse (`RMSE ~1.31 uT`, `corr ~0.47`)
- `fold symmetric / fit native`: worst (`RMSE ~1.74 uT`, `corr ~0.07`)
- `fold symmetric / fit symmetric`: still worse than native/native
  (`RMSE ~1.13 uT`, `corr ~0.24`)

### Chi2 comparison

`chi2` was not discriminative in this study:

- normal fits (`native` vs `symmetric`): nearly identical (`~1e-5` mean/p95),
- folded fits across modes: similarly tiny values.

### Important nuance

Symmetric folding reduced fold residual strongly (roughly `0.225 -> 0.029` mean),
but that did not translate to better folded-vs-normal B111 agreement.

## Decision from this investigation

- Do **not** implement symmetric range restriction as a default at this time.
- Keep this QEP on hold pending a revised design that improves fit agreement,
  not just fold residual.
- Continue focusing on fold-center estimator quality and folded-initial-guess
  strategy (tracked in QEP-063 follow-up work).

## Design

Add optional folded pre-step: range symmetrization before fold-center estimation.

### 1) Symmetrization modes

- `none` (current behavior)
- `trim_only`
  - enforce mirrored low/high support around a chosen `D_ref`
  - keep branch content unchanged aside from mirrored support selection
- `recenter_and_trim`
  - resample both branches onto mirrored offsets around `D_ref`
  - use symmetric support only

### 2) Reference center (`D_ref`) modes

- `d_zfs_nominal` (`D_ZFS` constant)
- `mean_spectrum_center` (per polarity global center from mean low/high spectra)
- `auto` (default, starts with `mean_spectrum_center`)

### 3) Fold-center estimator remains orthogonal

Estimator choices (`bruteforce`, `centroid`, `centroid_p2`, FFT variants) remain
independent. Symmetrization preconditions the inputs to center estimation.

## Implementation Plan (Deferred)

1. Preserve prototype script and measurements as reference evidence.
2. Re-open only after a revised design shows improvement on 15N fixtures under
   folded-vs-normal B111 metrics.
3. If reopened, require both 15N gains and 14N non-regression before merging.

## Testing Strategy

Real fixtures (128x128):

- 15N: `real_fov18x_fov5838_x78y24`, `real_fov18x_fov7539_x99y31`,
  `real_fov18x_fov14925_x45y62`
- 14N: `real_fov1_fov2037485_x365y1061`, `real_fov1_fov1069295_x1775y556`,
  `real_fov1_fov1074295_x1015y559`

Acceptance checks:

- 15N: improve folded-vs-normal induced RMSE/correlation versus current default,
- 14N: no meaningful regression versus current default,
- runtime overhead remains acceptable for large FOVs.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Over-trimming removes informative spectral region | Track retained span and fallback to `none` when too short |
| 15N-tuned behavior regresses 14N | Keep both fixture families in regression gates |
| Interpolation artifacts from recentering | Evaluate fold residual and fit metrics together |

## GUI Integration Requirements

- No GUI API break required if defaults remain internal and automatic.
- Optional future GUI advanced control can expose symmetrization mode.
- Units, map conventions, and result schema remain unchanged.

## Out of Scope

- Redesign of folded model parameterization.
- Changes to normal (non-folded) fit path.
- New serialization/output formats.
