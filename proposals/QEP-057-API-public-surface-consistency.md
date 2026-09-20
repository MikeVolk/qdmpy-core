# QEP-057-API - Public Surface Consistency and Deprecation Policy

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-042, QEP-044, QEP-051-ARCH |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-09 |

---

## Motivation

The top-level API is strong, but user-facing consistency has drifted in a few
places:

1. "Pure container" intent vs convenience wrappers can be confusing.
2. Some docstrings describe behavior/defaults that no longer match runtime.
3. Alias-heavy paths increase cognitive load for users and contributors.

This QEP defines one canonical surface and a clear deprecation policy.

---

## Goals

- Define canonical APIs for load/fit/save/plot workflows.
- Align docs/docstrings with runtime defaults and behavior.
- Keep backward compatibility via explicit deprecations, not silent drift.
- Improve discoverability through consistent naming and examples.

## Non-goals

- No immediate removal of convenience wrappers.
- No package namespace rename.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### 1) Canonical path table

| Task | Canonical API |
|---|---|
| Load measurement | `qdmpy.load(...)` or `Measurement.from_folder(...)` |
| Fit | `Measurement.fit_odmr(...)` / `Measurement.fit_folded_odmr(...)` |
| Persist | `qdmpy.io.save_qdm(...)` / `qdmpy.io.load_qdm(...)` |
| Plot | `qdmpy.plotting.*` functions |

### 2) Wrapper policy

- Keep wrappers for one release cycle.
- Mark as convenience aliases in docs and runtime warnings where appropriate.
- Add explicit removal-target metadata in proposal/docs.

### 3) Defaults and terminology alignment

- Audit entrypoint docstrings for default values and behavior language.
- Ensure normalization messaging consistently reflects current mean-normalized
  default and max deprecation state.

---

## Files to Change

| File | Change |
|------|--------|
| `src/qdmpy/__init__.py` | Entrypoint docstring/default wording alignment |
| `src/qdmpy/result.py` | Clarify wrapper status and deprecation messaging |
| `docs/migration.md` | Canonical-path guidance and deprecation timeline |
| `docs/quickstart.md` | Ensure examples use canonical paths |

---

## Test Plan

- Public API import smoke tests still pass.
- Doc-example snippets execute against current signatures.
- Contract tests assert key defaults documented == runtime defaults.

---

## Acceptance Criteria

- Canonical API table is documented and reflected in tutorials.
- No contradictory default descriptions in key entrypoints.
- Deprecated aliases are explicit and time-bounded.
