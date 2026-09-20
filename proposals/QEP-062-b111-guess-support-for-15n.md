# QEP-062 -- B111 Guess Support for 15N

| Field   | Value |
|---------|-------|
| Status  | Draft |
| Created | 2026-03-10 |
| Scope   | `qdmpy.odmr.analysis.b111_from_dip_positions` |
| Depends | QEP-GUI-013 (consumer behavior in GUI) |

## Motivation

`b111_from_dip_positions()` provides a fast, pre-fit B111 estimate used by the
GUI (`B111 guessed`). Today this path is effectively tuned for 14N behavior and
does not reliably support 15N datasets.

As a result, GUI users can see missing or incorrect guessed B111 maps for 15N,
despite having valid ODMR data. This creates inconsistency between diamond
types and undermines confidence in pre-fit diagnostics.

## Design

### 1. Define explicit contract for guessed B111 across diamond types

`b111_from_dip_positions(data)` should either:

- return a physically valid `{'remanent': ..., 'induced': ...}` estimate, or
- raise a specific, user-actionable `DataValidationError` describing why the
  estimate is unavailable for the provided data/model.

Silent wrong outputs are not acceptable.

### 2. Add 15N-aware dip position extraction path

Refactor dip selection logic to support both 14N and 15N input patterns while
preserving the same output units and sign conventions:

- internal frequencies remain GHz,
- output maps remain uT,
- polarity/frequency labels remain `neg/pos` and `low/high`.

Implementation may infer strategy from dataset metadata when present, with a
safe fallback heuristic when metadata is absent.

### 3. Harden validation and diagnostics

Improve validation so failures identify root cause (for example: missing coords,
unsupported branch structure, or ambiguous minima). Provide deterministic error
messages suitable for GUI display.

## Implementation Plan

1. Document expected dip-selection behavior for 14N vs 15N in docstrings/tests.
2. Refactor `b111_from_dip_positions()` internals into small strategy helpers.
3. Add 15N handling path and route by metadata/heuristic.
4. Add/extend synthetic fixtures and tests for both 14N and 15N.
5. Verify old 14N outputs remain unchanged within numeric tolerance.
6. Update `CHANGELOG.md` under `## [Unreleased]`.

## Testing

- Unit tests for validation failures (missing labels, malformed dimensions).
- Golden-path tests for 14N and 15N synthetic data.
- Regression test that 14N behavior remains stable.
- Property checks for shape/unit/sign invariants of returned maps.

## Risks

| Risk | Mitigation |
|------|------------|
| Incorrect 15N branch inference from limited metadata | Prefer explicit metadata, use conservative fallback, fail clearly on ambiguity |
| Regressions in existing 14N behavior | Keep dedicated 14N regression tests and tolerance checks |
| Overly permissive heuristics producing plausible but wrong maps | Favor explicit failure over weak inference |

## GUI Integration Requirements

- API contract stays unchanged: `b111_from_dip_positions(data) -> {'remanent', 'induced'}`.
- On supported 15N datasets, GUI `B111 guessed` maps should render normally.
- On unsupported/ambiguous datasets, core must raise `DataValidationError` with
  message suitable for GUI error display; GUI should not show stale guessed maps.
- No GUI-side unit conversion changes required.
- Acceptance check from GUI perspective:
  - load 14N dataset -> guessed maps available,
  - load 15N dataset -> guessed maps available (or explicit error, no silent misuse).

## Out of Scope

- Changing fitted B111 computation.
- Introducing new public API specifically for guessed-B111 mode selection.
- GUI copy/layout changes (handled in QEP-GUI-013 and follow-ups).
