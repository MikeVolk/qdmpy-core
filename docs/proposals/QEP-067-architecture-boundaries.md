# QEP-067: Re-establish architectural boundaries around workflows, results, persistence, and configuration

| Field | Value |
|---|---|
| Status | Draft |
| Author | OpenCode |
| Date | 2026-03-22 |
| Revised | 2026-03-22 - narrowed scope after code audit; QDMResult already aligned, remaining work focused on FitResult, MagneticMap, Measurement, settings flow, and `.qdm` subtype loading; 2026-03-22 - implemented remaining scope, recorded plotting extraction as already complete, moved MagneticMap persistence behind qdmpy.io, added explicit settings seams, and extracted concrete Measurement workflow helpers |
| Scope | Large |
| Type | Architecture / phased refactor |

## Summary

The repository claims a layered, immutable design, but several important boundaries still do not hold in practice. The remaining issues are concentrated in workflow orchestration, plotting and persistence on selected result objects, hidden configuration access, and `.qdm` persistence losing field-source subtype information.

This proposal restores the intended dependency direction incrementally: edge concerns move toward explicit adapters and services, domain and result types stay stable, and the current public API remains available through thin compatibility wrappers during migration. `QDMResult` is already mostly aligned with that target and is not the main focus of the remaining work.

## Current state

- `Measurement` is the dominant orchestration object in `src/qdmpy/measurement.py`. It handles folder loading, metadata resolution, image fallback, processor assembly, fitting, refitting, folding, and display.
- `QDMResult` is already largely aligned with the intended boundary: it documents itself as a pure data container and keeps only thin `save()` / `load()` wrappers over `qdmpy.io`.
- `FitResult`, `MagneticMap`, and `Measurement` plotting methods are already thin delegates into `src/qdmpy/plotting/*`; plotting extraction is no longer a remaining migration step.
- `MagneticMap` persistence now delegates through `src/qdmpy/io/magnetic_map.py`, keeping `src/qdmpy/magnetic_map.py` as a thin wrapper.
- `Measurement` remains the main public workflow object, but its folder-loading, fit/refit, and folding orchestration now delegate through `src/qdmpy/measurement_workflows.py` instead of owning all implementation details directly.
- Core algorithms depend on global settings and environment-derived state:
  - `src/qdmpy/fitting/manager.py`
  - `src/qdmpy/magnetic_map.py`
  - `src/qdmpy/settings.py`
- `Measurement.fit_odmr()`, `Measurement.fit_folded_odmr()`, `Measurement.refit_outliers()`, and `MagneticMap.from_b111()` now accept explicit settings/dependency inputs so tests and callers can avoid ambient global state on the identified seams.
- The `.qdm` persistence layer serializes field-source kind information, but deserializes everything as the base `FieldSource` in `src/qdmpy/io/qdm.py`.
- The top-level API in `src/qdmpy/__init__.py` is broad and treated as a compatibility contract by `tests/test_imports.py`, so boundary cleanup must preserve compatibility wrappers.

## Motivation / problem statement

The current structure contradicts the repo's own architectural claims in `docs/migration.md` and prior cleanup work referenced in `CHANGELOG.md` for QEP-036, QEP-038, QEP-041, and QEP-042.

The main risks are structural rather than stylistic:

- selected result and workflow objects are still coupled to plotting and file formats
- configuration is pulled implicitly into core operations
- persistence does not preserve domain subtype semantics
- broad orchestration hubs keep attracting unrelated responsibilities

If this continues, local changes to plotting, persistence, settings, or workflow UX will keep forcing edits across core modules and blocking cleaner reuse. The main remaining hotspots are `Measurement`, `FitResult`, `MagneticMap`, and `.qdm` deserialization.

## Goals / non-goals

### Goals

- restore one-way dependency direction from edges toward core logic
- preserve current public behavior initially through thin wrappers
- fix the broken field-source persistence seam
- make configuration dependencies explicit in core services
- reduce coordination pressure on `Measurement` and `FitManager`

### Non-goals

- no big-bang rewrite
- no immediate removal of convenience methods from public objects
- no immediate contraction of the top-level `qdmpy` API
- no redesign of fitting physics, processor semantics, or file formats beyond boundary cleanup

## Proposed design

Introduce an explicit service and adapter layer for workflow and edge concerns while retaining current convenience methods as delegating wrappers.

Keep domain and result types focused on validated data and domain transformations:

- `ODMRData`
- `FitResult`
- `QDMResult`
- `MagneticMap`
- field-source models

`QDMResult` already follows this direction closely and should be treated as the reference boundary for result-container behavior: no plotting methods, and only thin persistence wrappers.

Shift edge concerns into explicit modules:

- workflow services for load/process/fit/fold orchestration
- persistence adapters for `.qdm` and `.npz`
- plotting adapters for result and measurement views
- configuration resolution at the application boundary rather than inside core algorithms

Preserve `qdmpy.load()`, `QDMResult.save()`, `QDMResult.load()`, `MagneticMap.display()`, and similar conveniences as thin wrappers during the transition. Where a wrapper already exists and is thin, keep it; where a method still owns real edge logic, move that logic out first and keep the method as a delegate.

Fix `.qdm` deserialization to reconstruct the discriminated `FieldSourceType` union instead of the base `FieldSource`.

## Decisions

### Decision 1: Keep compatibility wrappers during migration

**Context**

The current public API is broad and explicitly tested.

**Decision**

Maintain convenience methods and top-level exports in the near term, but reimplement them as thin delegates to adapter and service modules.

**Consequences**

- users keep the current ergonomics
- internal architecture can improve without an immediate breaking change

**Rejected alternatives**

- immediate API removal
- keeping the current direct coupling indefinitely

### Decision 2: Treat persistence and plotting as edge adapters, not domain responsibilities

**Context**

`QDMResult`, `FitResult`, `MagneticMap`, and `Measurement` are not all in the same state. `QDMResult` already behaves mostly as intended, while `FitResult`, `MagneticMap`, and `Measurement` still own substantive plotting or persistence behavior.

**Decision**

Move substantive save, load, and display logic into dedicated modules. Object methods remain thin delegation points only. Existing thin wrappers on `QDMResult` are kept, not reworked.

**Consequences**

- core modules stop owning file-format and visualization behavior where they still do today
- testing becomes more boundary-specific

**Rejected alternatives**

- keeping all convenience logic embedded in result objects
- removing all convenience methods at once

### Decision 3: Make configuration explicit in core services

**Context**

`FitManager` and `MagneticMap` currently read process-global settings internally.

**Decision**

Core paths should accept resolved settings or concrete parameters from callers. `get_settings()` remains an edge concern used by CLI, top-level helpers, and wrappers.

**Consequences**

- less hidden state
- easier testing
- fewer filesystem side effects in core code

**Rejected alternatives**

- keeping global singleton access in core algorithms
- passing raw environment and config concerns deep into domain code

### Decision 4: Preserve field-source subtype fidelity across persistence

**Context**

`.qdm` stores subtype data but discards it on load.

**Decision**

`.qdm` load must reconstruct the discriminated union defined in `src/qdmpy/field_source.py`.

**Consequences**

- source fitting and downstream behavior remain valid after round-trip persistence

**Rejected alternatives**

- normalizing all sources to a base type
- leaving subtype reconstruction to callers

## Alternatives considered

### Minimal bug-fix only

Fix `.qdm` subtype loading and leave broader architecture alone.

Rejected because it addresses one correctness issue but leaves the main boundary drift untouched.

### Full rewrite into strict clean architecture

Rejected because the repo already has functioning workflows and a strong public API contract. A rewrite adds migration risk without proportional benefit.

### Immediate API narrowing

Rejected because it would break established imports and tests before the internal seams are ready.

## Implementation steps

### Step 1: Repair the persistence boundary

- Goal: preserve `FieldSourceType` round trips for `.qdm`
- Affected files: `src/qdmpy/io/qdm.py`, `src/qdmpy/field_source.py`, targeted persistence tests
- Dependencies: none
- Validation: `.qdm` round-trip tests for `MagneticSource` and `UpwardContinuedSource` in addition to the existing generic `FieldSource` coverage
- Status: completed on 2026-03-22

### Step 2: Extract plotting adapters behind existing convenience methods

- Goal: keep methods like `display()` and `plot()` public, but move real behavior out of core and result modules that still own it
- Affected files: `src/qdmpy/fitting/result.py`, `src/qdmpy/magnetic_map.py`, `src/qdmpy/measurement.py`, `src/qdmpy/plotting/*`
- Dependencies: independent of step 1
- Validation: no behavioral change in plotting smoke tests; convenience methods still work; `QDMResult` remains free of plotting methods
- Status: already complete before implementation; proposal drift confirmed during audit on 2026-03-22

### Step 3: Extract persistence adapters behind existing convenience methods

- Goal: complete persistence boundary cleanup for objects that still own direct file-format behavior
- Affected files: `src/qdmpy/magnetic_map.py`, `src/qdmpy/io/*.py`
- Dependencies: none, but aligns with step 2
- Validation: existing NPZ and QDM round-trip tests still pass; `QDMResult.save()` / `load()` remain thin wrappers
- Status: completed on 2026-03-22 by moving `MagneticMap.save()` behind `qdmpy.io.save_magnetic_map()`

### Step 4: Move configuration resolution outward

- Goal: stop core algorithm paths from pulling `get_settings()` implicitly where practical
- Affected files: `src/qdmpy/fitting/manager.py`, `src/qdmpy/magnetic_map.py`, `src/qdmpy/settings.py`, top-level wrappers and CLI paths
- Dependencies: steps 2 and 3 make this easier
- Validation: tests can inject explicit settings and parameters without touching global process state
- Status: completed on 2026-03-22 for the identified seams by threading explicit settings and GPU availability through `Measurement` wrappers and `MagneticMap.from_b111()` while preserving compatible defaults

### Step 5: Decompose workflow orchestration

- Goal: reduce `Measurement` responsibility by extracting loading, fitting, and folding orchestration services
- Affected files: `src/qdmpy/measurement.py`, possible new workflow modules, related tests and docs
- Dependencies: steps 2 through 4
- Validation: `qdmpy.load()` and `Measurement.from_folder()`, `fit_odmr()`, and `fit_folded_odmr()` preserve current observable behavior
- Status: completed on 2026-03-22 by extracting concrete helpers into `src/qdmpy/measurement_workflows.py` while keeping `Measurement` methods as thin public wrappers

## Risks and tradeoffs

- Keeping thin wrappers means some coupling remains temporarily by design.
- The top-level API contract slows aggressive cleanup. That is acceptable because stability matters more here.
- Refactoring settings flow can expose hidden assumptions in tests and wrappers.
- Splitting orchestration too aggressively would create service sprawl, so extraction should follow concrete seams only.

## Migration / rollout plan

- Phase 1: correctness-first boundary repairs with no public API change
- Phase 2: adapter extraction behind existing object methods
- Phase 3: explicit configuration injection for core paths
- Phase 4: workflow-service decomposition and optional later proposal for API narrowing

Any public deprecations should be deferred until wrappers have been stable for at least one release cycle.

## Validation / testing plan

Preserve existing regression coverage for imports, CLI, persistence, and result behavior:

- `tests/test_imports.py`
- `tests/test_cli.py`
- `tests/test_qdm_result.py`
- `tests/test_io_qdm.py`

Add new tests for:

- `.qdm` subtype round-tripping of `MagneticSource` and `UpwardContinuedSource`
- convenience methods delegating through adapter boundaries
- explicit-settings execution paths that avoid global singleton dependence

Done should include unchanged overall suite health, plus new targeted coverage on the repaired seams.

## Acceptance criteria

### Observable behavior changes

- users can still call the current convenience API without regressions
- `.qdm` files preserve concrete field-source types on load
- plotting and persistence still work through public wrappers
- core execution paths can be exercised with explicit settings and parameters in tests

### Tests and validation

- existing import, CLI, result, and persistence tests pass
- new round-trip tests for `FieldSourceType` persistence pass
- existing tests that enforce `QDMResult` as a no-plot, thin-wrapper boundary still pass
- new tests verify wrappers delegate correctly without re-embedding edge logic

### Done looks like

- substantive plotting and persistence logic no longer lives directly in `FitResult`, `MagneticMap`, and `Measurement`
- configuration access is pushed outward on the identified core paths
- `Measurement` no longer owns every workflow concern directly
- no unnecessary new abstraction layer exists beyond concrete adapter and service seams

## Implementation-fit checklist

- The implementation preserves current public wrappers while moving real work behind them.
- The `.qdm` loader reconstructs discriminated field-source subtypes correctly.
- Core modules do not gain new plotting, persistence, or framework responsibilities.
- Settings resolution is more explicit after the change, not more ambient.
- `Measurement` and `FitManager` are smaller in responsibility, not just split into more indirection.
- Any deferred API cleanup is documented explicitly rather than smuggled into this refactor.

## Open questions

- Whether a later follow-up QEP should narrow the top-level `qdmpy` API after the internal seams are stable.
