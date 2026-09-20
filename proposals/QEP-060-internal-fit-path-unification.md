# QEP-060 - Internal Fit Path Unification for Folded and Non-Folded Workflows

| Field | Value |
|-------|-------|
| **Status** | Superseded by QEP-070 |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-059 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-10 |

---

> Superseded 2026-07-05 by QEP-070 (fit-pipeline-unification), which implements this
> unification together with QEP-FIT-003's decomposition since removing `fit_folded`'s
> second-`FitManager` construction requires the same non-mutating constraint mechanism
> the decomposition introduces.

---

## Motivation

QEP-059 unified the user-facing constraint model and result type, but fitting
still exposes two private execution paths:

- `FitManager.fit(...)` for standard ODMR data.
- `FitManager.fit_folded(...)` for folded ODMR data.

This leaves room for contract drift in:

1. auto-model detection inputs (`raw_data` forwarding and fallback behavior),
2. error behavior and validation messages,
3. future changes to constraint handling and result assembly.

Users should keep both workflows (`fit_odmr` and `fit_folded_odmr`), but core
internals should execute through one shared fit pipeline.

---

## Goals

- Keep both public workflows unchanged:
  - `Measurement.fit_odmr(...)`
  - `Measurement.fit_folded_odmr(...)`
- Unify private fitting execution into one internal path.
- Ensure folded fitting is only a data-preparation adaptation step.
- Eliminate drift-prone duplicated logic for model resolution, constraints, and
  result construction.

## Non-goals

- No removal or rename of public folded APIs.
- No change to folded spectrum construction itself (QEP-011 scope).
- No physics-convention changes (units, sign conventions, B111 equations).
- No `.qdm` format changes.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

---

## Design

### 1) Shared internal fit pipeline

Introduce a single private method in `FitManager` (name illustrative):

```python
def _fit_prepared(
    self,
    prepared_data: xr.DataArray,
    prepared_freq_ghz: NDArray,
    *,
    pixel_spacing: float,
    detection_data: NDArray | None,
) -> FitResult:
    ...
```

Responsibilities:

- validate prepared inputs,
- resolve model in auto mode from `detection_data` policy,
- apply constraints through one path,
- run backend fit,
- construct `FitResult` with shared semantics.

### 2) Folded path becomes preparation only

Folded-specific operations remain only in a preparation adapter:

- convert folded `delta_f_ghz` axis to absolute GHz,
- reshape to the standard fitting tensor shape,
- define detection input (raw unfolded data if model is auto).

After preparation, folded and non-folded both call `_fit_prepared(...)`.

### 3) Keep public API distinction

- `Measurement.fit_folded_odmr(...)` remains as the explicit folded workflow.
- `Measurement.fit_odmr(...)` remains unchanged.

The distinction is product-level and user-level, not solver-contract-level.

### 4) `fit_folded` disposition

Recommended option: retain `fit_folded(...)` as a thin adapter that only does
folded preparation and immediately delegates to `_fit_prepared(...)`.

Rationale:

- minimal public/internal call-site churn,
- preserves readability at the measurement layer,
- prevents accidental reintroduction of divergent solver behavior.

---

## Implementation Plan

### Phase 1 - Extract shared execution

1. Introduce `_fit_prepared(...)` in `FitManager`.
2. Route `fit(...)` through `_fit_prepared(...)` after standard preparation.
3. Keep behavior identical; add parity tests for unchanged outputs.

### Phase 2 - Route folded through shared execution

4. Refactor `fit_folded(...)` to preparation + delegation only.
5. Ensure `Measurement.fit_folded_odmr(...)` forwards unfolded `raw_data` when
   model auto-detection is used.
6. Keep explicit `model_name` flow bypassing auto detection.

### Phase 3 - Drift guardrails

7. Add tests asserting identical constraint semantics for both paths.
8. Add tests for consistent error messages and exception types.
9. Document the unified internal contract in developer docs.

---

## Test Plan

- Unit tests:
  - folded and non-folded both call shared private execution path,
  - auto-model detection behavior is deterministic and policy-compliant,
  - explicit `model_name` bypasses detection in both flows.
- Regression tests:
  - folded and non-folded outputs preserve existing shape/unit contracts,
  - folded B111 values remain within expected tolerance vs current behavior.
- GUI smoke checks:
  - run folded and non-folded fit actions from GUI,
  - verify maps render and save/reload workflow remains stable.

---

## Risks and Mitigations

1. **Behavior drift in folded edge cases**
   - Mitigation: fixture-based regression tests and explicit parity assertions.
2. **Private-method overcoupling**
   - Mitigation: keep preparation and execution boundaries strict and typed.
3. **False confidence from unit-only validation**
   - Mitigation: include GUI smoke checks in acceptance criteria.

---

## Acceptance Criteria

- Both public workflows remain available and documented.
- One internal fit execution path handles model resolution, constraints, and
  result assembly for both folded and non-folded inputs.
- `fit_folded(...)` contains no unique solver-contract logic beyond folded
  preparation.
- Tests cover parity, detection behavior, and error contract consistency.
- GUI folded/non-folded workflows run without GUI-side workaround logic.

---

## Alternatives Considered

1. **Remove `fit_folded(...)` entirely**
   - Pros: fewer methods.
   - Cons: more churn in call sites and reduced readability for folded workflow.

2. **Keep current split and rely only on tests**
   - Pros: lowest immediate effort.
   - Cons: preserves structural drift risk and duplicated logic surface.
