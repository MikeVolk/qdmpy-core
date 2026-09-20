# QEP-055-UX - CLI UX Consistency and Command Contracts

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Priority** | P2 |
| **Complexity** | M |
| **Depends on** | QEP-051-ARCH |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-09 |

---

## Motivation

The CLI is intentionally small, but currently inconsistent with project identity
and user expectations:

1. Version lookup can return `unknown` due to distribution-name mismatch.
2. Help and error outputs are not treated as stable UX contracts.
3. The `models` command output is useful but does not guarantee deterministic
   ordering/format for scripting.

This creates friction for first-time users, CI scripts, and reproducibility.

---

## Goals

- Make `qdmpy --version` deterministic and correct.
- Define command-level UX contracts (stdout/stderr/exit code).
- Keep output stable enough for simple parsing and automation.
- Improve discoverability without expanding CLI scope dramatically.

## Non-goals

- No full pipeline execution CLI in this QEP.
- No replacement of `argparse`.
- No breaking rename of existing `models` command.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### 1) Version identity contract

- Resolve version from distribution `qdmpy-core`.
- Fallback order:
  1. `importlib.metadata.version('qdmpy-core')`
  2. `qdmpy.__version__`
  3. literal `unknown`

### 2) Stable command contract table

| Command | Success exit | Failure exit | Output contract |
|---|---:|---:|---|
| `qdmpy --version` | 0 | n/a | one-line `QDMpy v<version>` |
| `qdmpy` (no subcommand) | 2 | n/a | help text printed |
| `qdmpy models` | 0 | n/a | deterministic list order |
| `qdmpy models <bad>` | n/a | 1 | clear error + available models |

### 3) `models` output consistency

- Sort model names consistently.
- Keep one model per line in non-detailed mode.
- Keep parameter/unit block in deterministic parameter order in detailed mode.

### 4) Error-message ergonomics

- For unknown model names, include nearest-name suggestion when possible.
- Avoid stack traces unless `--debug` is set.

---

## Files to Change

| File | Change |
|------|--------|
| `src/qdmpy/cli/__init__.py` | Version resolution and entrypoint UX contract |
| `src/qdmpy/cli/qdmpy_cli.py` | Deterministic output and unknown-model messaging |
| `tests/test_cli.py` | Add/expand contract tests for version/help/errors |

---

## Test Plan

- `qdmpy --version` prints non-`unknown` in package context.
- `qdmpy` with no command exits `2` and shows help.
- `qdmpy models` output is deterministic.
- Unknown model returns `1` with available-model context.
- `--debug` enables traceback; non-debug path does not.

---

## Acceptance Criteria

- CLI version reflects installed distribution identity.
- Command outputs/exit codes are documented and enforced by tests.
- No regressions in existing command behavior.
