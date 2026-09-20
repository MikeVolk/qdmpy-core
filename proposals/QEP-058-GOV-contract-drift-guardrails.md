# QEP-058-GOV - Contract Drift Guardrails in CI

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-051-ARCH, QEP-057-API |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-09 |

---

## Motivation

Several regressions were not algorithmic failures but contract drift between:

- docs and runtime behavior,
- type hints and runtime acceptance,
- API argument promises and actual call flow.

We need lightweight CI guardrails that fail fast when contracts drift.

---

## Goals

- Add automated checks for critical API/UX contracts.
- Prevent default-value and behavior drift in key entrypoints.
- Keep checks lightweight and maintainable.

## Non-goals

- No broad snapshot testing of full docs site.
- No static-analysis replacement for existing ruff/ty checks.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Contract Categories

1. **Signature contracts**
   - key function parameters and defaults remain expected.
2. **Behavior contracts**
   - critical args are honored end-to-end (e.g. folded constraints).
3. **Persistence contracts**
   - save/load preserves required fields.
4. **Identity contracts**
   - package/CLI version and naming remain consistent.

---

## Design

### 1) Contract test module

- Add `tests/contracts/` with explicit invariant-focused tests.
- Keep tests narrow and high-signal.

### 2) Drift linting helpers

- Optional helper checks for repeated known drift vectors:
  - normalization wording mismatch,
  - stale package naming (`QDMpy` vs `qdmpy-core`) in executable paths.

### 3) CI integration

- Run contract tests in default PR CI.
- Gate merges on contract suite success.

---

## Files to Change

| File | Change |
|------|--------|
| `tests/contracts/test_entrypoint_contracts.py` | New |
| `tests/contracts/test_persistence_contracts.py` | New |
| `.github/workflows/*.yml` | Ensure contract suite runs in CI |

---

## Test Plan

- Contract tests fail on intentional drift.
- Contract tests pass on current intended behavior.
- CI runtime impact remains acceptable.

---

## Acceptance Criteria

- Contract suite exists and is part of required CI checks.
- At least one test per contract category is implemented.
- Recent drift classes are explicitly covered by tests.
