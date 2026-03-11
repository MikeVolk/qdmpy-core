# QEP-059 - Unified Constraint Interface with mT Units

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | QEP-051 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-10 |
| **Revised** | 2026-03-10 |

---

## Motivation

qdmpy-core currently has two completely separate constraint and fitting paths:

1. **Non-folded (standard):** Constraints in absolute GHz (e.g. center 2.0--3.1 GHz).
   The optimizer fits absolute resonance frequencies. B111 is computed by
   subtracting D_ZFS from the fitted center, then dividing by gamma_NV.

2. **Folded:** Constraints in delta-f GHz (e.g. center 0.001--0.080 GHz),
   hardcoded in `_FOLDED_*` module constants. A separate `FoldedFitResult`
   subclass overrides the B111 calculation to skip the D_ZFS subtraction.

This dual-path design causes:

- Two constraint tables with no shared source of truth
- A `FoldedFitResult` subclass that only exists to override one method
- Constraints expressed in abstract GHz values that have no physical intuition
  for users (what does `center_max = 3.1 GHz` mean in terms of field?)

### Experimental findings

Comparison of absolute-GHz vs delta-f fitting on real data
(`scripts/compare_nonfolded_delta_parameterization.py`, FOV1, ESR14N, bin4,
full FOV 300x480):

- **Centers/B111 are equivalent:** differences < 0.02 MHz / 0.35 uT
- **Absolute-GHz fitting has ~1% lower chi2** uniformly across the FOV
- **Both converge fully** on all 144,000 pixels (zero max-iteration hits)

The chi2 advantage of absolute-GHz fitting is small but systematic. The
Levenberg-Marquardt optimizer benefits from the larger parameter scale
(center ~2.87 GHz vs ~0.025 GHz) which gives better-conditioned Jacobians.

**Conclusion:** Keep absolute GHz as the internal fitting domain for all paths
(including folded spectra), but give users a physical-units constraint
interface in millitesla.

---

## Goals

- Introduce **mT-based constraints** as the default user-facing interface,
  settable in `QDMpySettings`.
- Support **absolute GHz constraints** as an alternative mode for power users.
- **Unify the folded and non-folded fitting paths**: all fitting uses
  absolute-GHz frequencies internally, one `FitResult` class, one B111
  calculation.
- **Remove `FoldedFitResult`** (not deprecate -- delete).
- Remove hardcoded `_FOLDED_*` constraint constants.

## Non-goals

- No change to the pyGpufit ESR kernels or model_id values.
- No change to physics conventions (sign, polarity, gamma_NV).
- No change to `.qdm` on-disk format.
- No change to the spectral folding pre-processing step itself (QEP-011).


## Design

### Core idea: mT constraints, absolute-GHz fitting

Users specify constraints in millitesla (Zeeman shift). Internally, these are
converted to absolute GHz for the optimizer. The folded path shifts its
delta-f frequency axis back to absolute before fitting, so everything goes
through the same code.

### 1) Settings: `constraint_units` mode

```python
class ModelConstraintsSettings(BaseModel):
    constraint_units: Literal['mt', 'absolute_ghz'] = 'mt'

    # -- mT mode (default) --
    center_max_mt: float = 6.0       # max Zeeman shift in mT
    center_min_mt: float = 0.0       # min Zeeman shift in mT (usually 0)
    width_max_mt: float = 0.7        # max linewidth in mT
    width_min_mt: float = 0.004      # min linewidth in mT

    # -- absolute GHz mode (power users / backward compat) --
    center_min: float = 2.0          # absolute GHz
    center_max: float = 3.1          # absolute GHz
    width_min: float = 0.0001        # GHz
    width_max: float = 0.005         # GHz

    # -- shared (unitless, same in both modes) --
    center_type: Literal['FREE', 'LOWER', 'UPPER', 'LOWER_UPPER'] = 'LOWER_UPPER'
    width_type: Literal['FREE', 'LOWER', 'UPPER', 'LOWER_UPPER'] = 'LOWER_UPPER'
    contrast_min: float = 0.003
    contrast_max: float = 0.0        # 0 = no upper bound
    contrast_type: Literal['FREE', 'LOWER', 'UPPER', 'LOWER_UPPER'] = 'LOWER'
    offset_min: float = 0.0
    offset_max: float = 0.0
    offset_type: Literal['FREE', 'LOWER', 'UPPER', 'LOWER_UPPER'] = 'FREE'
```

The TOML config looks like:

```toml
[model.constraints]
constraint_units = "mt"      # or "absolute_ghz"
center_max_mt = 7.0          # ~7 mT default safety margin for larger bias scans
width_max_mt = 0.7
```

**Files:** `src/qdmpy/settings.py`

### 2) Constraint conversion in ConstraintManager

`ConstraintManager.__init__` reads `constraint_units` and produces
**absolute-GHz optimizer bounds**.

**mT mode -> absolute GHz (all fits):**
```python
delta_min_ghz = center_min_mt * 1e-3 * GAMMA_NV
delta_max_ghz = center_max_mt * 1e-3 * GAMMA_NV

center_min_ghz = D_ZFS - delta_max_ghz
center_max_ghz = D_ZFS + delta_max_ghz

width_min_ghz = width_min_mt * 1e-3 * GAMMA_NV
width_max_ghz = width_max_mt * 1e-3 * GAMMA_NV
```

**absolute_ghz mode (all fits):** Use `center_min`/`center_max` and
`width_min`/`width_max` as-is.

When `center_min_mt > 0` and `center_type='LOWER_UPPER'`, `FitManager` applies
branch-specific center windows during per-range fitting so `2-7 mT` behaves as
a true window, not just an upper bound:

```python
# low-frequency branch
center in [D_ZFS - delta_max_ghz, D_ZFS - delta_min_ghz]

# high-frequency branch (and folded absolute-GHz branch)
center in [D_ZFS + delta_min_ghz, D_ZFS + delta_max_ghz]
```

There is no folded-specific optimizer domain; folded and non-folded both fit in
absolute GHz.

**Files:** `src/qdmpy/fitting/constraints.py`, `src/qdmpy/fitting/manager.py`

### 3) Folded fitting uses absolute-GHz domain

Currently `fit_folded()` passes delta-f frequencies to the optimizer. Instead:

```python
# In FitManager.fit_folded():
# Shift folded frequency axis back to absolute GHz
folded_freq_abs = D_ZFS + delta_f_ghz   # e.g. 2.871--2.950 GHz
```

Then call the standard `fit()` path with `folded_freq_abs`. The fitted
centers come back as absolute GHz (~2.87 + shift), and the existing
`FitResult._calc_delta_from_single_center` with `n_frange=1` does
`(center - D_ZFS) / GAMMA_NV * 1e6` to get B111 in uT.

This eliminates the need for `FoldedFitResult` entirely.

**Files:** `src/qdmpy/fitting/manager.py`

### 4) Remove FoldedFitResult

With folded fits producing absolute-GHz centers, the base `FitResult` handles
everything. Delete `FoldedFitResult` class.

**Files:** `src/qdmpy/fitting/result.py`

### 5) Remove _FOLDED_* constants

The hardcoded `_FOLDED_CENTER_MIN`, `_FOLDED_CENTER_MAX`, etc. are replaced
by the settings-driven constraint conversion. Delete them.

**Files:** `src/qdmpy/fitting/manager.py`

### 6) Remove for_folded()

`FitManager.for_folded()` is deleted. It existed solely to build a
FitManager with the hardcoded `_FOLDED_*` constraint table. With
constraints now driven by settings (mT or absolute GHz) and `fit_folded()`
handling the frequency-axis shift internally, there is no separate
construction path needed.

**Files:** `src/qdmpy/fitting/manager.py`

---

## Constraint conversion summary

| Setting mode | Non-folded fit | Folded fit |
|-------------|----------------|------------|
| `mt` (default) | mT -> absolute GHz via D_ZFS +/- delta | same absolute-GHz conversion |
| `absolute_ghz` | Use center_min/max as-is | same absolute-GHz bounds |

In both modes, the optimizer always receives absolute-GHz frequencies and
constraints. The only difference is how the user specifies the bounds.

---

## Implementation Plan

### Phase 1: Settings and constraint conversion

1. Add `constraint_units`, `center_max_mt`, `center_min_mt`, `width_max_mt`,
   `width_min_mt` fields to `ModelConstraintsSettings`.
2. Implement `constraint_units` handling in `ConstraintManager` with one
   internal output domain (absolute GHz).
3. Implement mT -> GHz conversion logic in `ConstraintManager`.
4. Apply per-branch center windows in `FitManager` when
   `center_min_mt > 0` and `center_type='LOWER_UPPER'`.
5. Keep absolute_ghz mode as pass-through bounds.
6. Tests: verify conversion math and branch-window behavior.

### Phase 2: Folded fitting in absolute domain

6. Modify `fit_folded()` to shift `delta_f_ghz` to `D_ZFS + delta_f_ghz`
   before fitting.
7. Return a standard `FitResult` (not `FoldedFitResult`).
8. Tests: verify folded B111 values match current results within tolerance.

### Phase 3: Remove FoldedFitResult and _FOLDED_* constants

9. Delete `FoldedFitResult` class from `result.py`.
10. Delete `_FOLDED_*` constants from `manager.py`.
11. Delete `for_folded()` classmethod.
12. Update all imports and references.
13. Tests: full test suite green.

### Phase 4: Cleanup and docs

14. Update `settings.toml` example in docs.
15. Add migration note: `center_min`/`center_max` in absolute GHz still work
    when `constraint_units = "absolute_ghz"`.
16. Update CHANGELOG.
17. Remove temporary developer validation script
    `scripts/compare_nonfolded_delta_parameterization.py` after implementation
    (results remain documented in this QEP and CHANGELOG).

---

## Experimental Evidence

### Real data: FOV1 (ESR14N, bin4, 300x480)

```
Center difference [MHz]: mean=3.4e-06, std=3.4e-04, max_abs=2.0e-02
Field difference [uT]:   mean=5.3e-05, std=8.5e-03, max_abs=3.5e-01
chi2 standard:    mean=1.3788e-06, median=1.3169e-06
chi2 delta-reparam: mean=1.3935e-06, median=1.3214e-06
Winner: standard (absolute GHz) -- ~1% lower chi2 uniformly
Convergence: 100% on all 576,000 fits (both methods)
```

The absolute-GHz parameterization gives marginally better chi2 due to
better-conditioned Jacobians at the ~2.87 GHz scale. Both methods produce
identical B111 maps for all practical purposes.

### Constraint range investigation

The original comparison used `center_max = 0.080 GHz` (~2.85 mT) for delta.
Re-running with 6.0 mT (~0.168 GHz, matching the absolute range) produced
identical results -- the chi2 difference is not caused by constraint clipping
but by the numerical scale of the optimizer parameters.

### Script (temporary)

`scripts/compare_nonfolded_delta_parameterization.py` was used to generate
the evidence in this QEP and is temporary. It should be removed in Phase 4
after implementation and verification are complete.

---

## Backward Compatibility

This QEP introduces an explicit **breaking change** in the fitting API surface.

- **Public API:** `fit_odmr()`, `FitResult`, `b111_remanent`, and
  `b111_induced` remain stable in behavior and values.
- **`FitResult.parameters["center"]`:** Remains in absolute GHz (~2.87).
  No semantic change. No version bump needed for this.
- **Settings:** New `constraint_units = "mt"` default. Users with existing
  `settings.toml` using `center_min`/`center_max` in absolute GHz must add
  `constraint_units = "absolute_ghz"` to preserve their values. Pydantic
  `extra = "ignore"` means old config files won't crash -- they just won't
  use the new mT fields unless `constraint_units` is set.
- **`FoldedFitResult`:** Removed. This is a **breaking change** because
  `FoldedFitResult` is currently exported from `qdmpy.fitting`.
  Code importing `FoldedFitResult` or checking `isinstance(..., FoldedFitResult)`
  will break and must migrate to `FitResult`.
- **`.qdm` files:** No format change. Centers remain absolute GHz.

---

## Risks and Mitigations

1. **Existing settings.toml with absolute GHz values**
   - Risk: Users who set `center_min = 2.70` in their config will silently
     get the new mT defaults instead.
   - Mitigation: If `center_min`/`center_max` are present in the TOML but
     `constraint_units` is absent, log a warning telling the user to add
     `constraint_units = "absolute_ghz"` or migrate to mT. Do not
     auto-migrate.

2. **Folded fit quality in absolute domain**
   - Risk: Shifting delta-f to absolute could change folded fit behavior.
   - Mitigation: The comparison script shows absolute is slightly better.
     Run folded-specific regression tests.

3. **mT defaults too wide/narrow for unusual samples**
   - Risk: 6 mT default may not cover high-field experiments.
   - Mitigation: It's a setting -- users can increase it. Document the
     conversion: `1 mT = 0.028 GHz delta-f`.

---

## Test Plan

- **Constraint conversion:** Unit tests for all 4 cases (mt/absolute x
  non-folded/folded) verifying GHz output values.
- **Regression:** Existing test suite passes with new defaults; B111 within
  rtol=1e-4.
- **Folded in absolute domain:** Fit folded data with shifted freq axis,
  compare B111 to current FoldedFitResult output.
- **Settings backward compat:** Load old settings.toml with absolute GHz
  values, verify warning logged and correct behavior.
- **Round-trip:** save/load FitResult preserves absolute-GHz centers.

---

## Acceptance Criteria

- `ModelConstraintsSettings` has `constraint_units` field defaulting to `'mt'`.
- mT constraints convert correctly to absolute GHz for both non-folded and
  folded paths.
- All fitting uses absolute-GHz frequencies internally.
- `FoldedFitResult` class is deleted.
- `_FOLDED_*` constants are deleted.
- `for_folded()` deleted.
- Full test suite green.
- `FitResult.parameters["center"]` is always absolute GHz.

---

## Resolved Questions

1. **Fitting domain:** Absolute GHz, not delta-f. Empirically ~1% better chi2
   and avoids changing the semantic meaning of `parameters["center"]`.
2. **User-facing units:** mT by default. Conversion is `delta_ghz = mt * 1e-3
   * GAMMA_NV` and `abs_ghz = D_ZFS +/- delta_ghz`.
3. **FoldedFitResult:** Delete, not deprecate. This is an intentional
   breaking change because the symbol is currently exported from
   `qdmpy.fitting`.
4. **Folded freq axis:** Shift to absolute (`D_ZFS + delta_f_ghz`) before
   fitting. Same optimizer, same FitResult, same B111 code path.
5. **center_max_mt = 7.0 mT default:** Slightly wider default safety margin for
   scans with larger induced field.
   The fitted center is the total Zeeman shift (remanent + bias projected
   onto [111]).
6. **Old settings.toml migration:** Log a warning, do not auto-migrate.
   Only one user currently has a custom config.
7. **`for_folded()`:** Remove entirely. No convenience wrapper needed since
   `fit_folded()` handles everything internally with settings-driven
   constraints.

## Open Questions

None.
