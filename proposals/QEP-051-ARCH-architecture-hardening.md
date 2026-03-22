# QEP-051-ARCH - Architecture Hardening and Boundary Cleanup

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | QEP-023, QEP-041, QEP-045, QEP-049 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-08 |
| **Amended** | 2026-03-09 |

---

## Motivation

`qdmpy-core` has a strong modular shape (ODMR -> fitting -> result -> I/O), but several architectural seams are still leaky:

1. Persistence is not fully faithful (`fit_states` is written to `.qdm` but not restored).
2. Folded-fit constraints are constructed in multiple places and caller-provided constraints can be lost.
3. Importing `qdmpy` performs global side effects (settings bootstrap + logger configuration).
4. Some extension contracts are inconsistent between docs, typing, and runtime checks.
5. A few boundary responsibilities are still mixed ("pure container" classes exposing convenience I/O).
6. Field source round-trip currently rebuilds base `FieldSource` objects rather than
   validating through the discriminated `FieldSourceType` union.
7. CLI/package identity has drifted (`QDMpy` vs `qdmpy-core`), causing version UX issues.
8. Critical API promises (especially folded constraints override behavior) are not asserted
   by explicit contract tests.

None of these issues requires a large redesign, but together they increase surprise, reduce confidence in round-trip reproducibility, and make extension behavior less predictable.

This QEP proposes targeted, low-risk hardening of boundaries and contracts.

---

## Goals

- Make persistence round-trips deterministic for fit diagnostics.
- Ensure folded and unfolded fitting use one consistent constraint path.
- Make `import qdmpy` side-effect-light and predictable.
- Align extension API promises with actual type/runtime behavior.
- Clarify layer boundaries without breaking the public API.
- Ensure field-source deserialisation preserves concrete subclass types.
- Ensure CLI reports the installed package version consistently.
- Add contract tests for high-risk UX/consistency invariants.

## Non-goals

- No change to QDM physics conventions.
- No rewrite of the fitting engine or ODMR pipeline.
- No breaking API removals in this QEP (deprecations allowed).

---

## Design

### 1) Persistence fidelity for `.qdm`

**Problem:** `save_qdm()` stores `fit/fit_states`, but `load_qdm()` drops it.

**Change:**
- Read `fit/fit_states` on load and restore as `FitResult.parameters['states']`.
- Keep behavior backward-compatible when the dataset is absent.
- Add a strict round-trip test asserting `states` equality for save/load.

**Files:**
- `src/qdmpy/io/qdm.py`
- `tests/...` (new/updated persistence tests)

---

### 2) Single folded-constraint source of truth

**Problem:** folded constraints are assembled more than once; `fit_folded()` currently spins up a second manager internally.

**Change:**
- Introduce one internal helper/factory for folded-domain constraints.
- Ensure `Measurement.fit_folded_odmr(..., constraints=...)` flows through to the effective folded fit manager.
- Keep `FitManager.for_folded()` as the canonical constructor for folded fits.
- Update docs to clearly describe override semantics.

**Files:**
- `src/qdmpy/fitting/manager.py`
- `src/qdmpy/measurement.py`
- docs/tutorial/API text where folded constraints are documented

---

### 3) Import-time side effects policy

**Problem:** `qdmpy.__init__` currently calls `get_settings()` unconditionally, creating config dirs and configuring loguru on import.

**Change:**
- Remove eager settings initialization from `__init__`.
- Keep lazy initialization via explicit `get_settings()` and runtime call sites that need it.
- Document initialization semantics clearly (what triggers config creation/logging setup).

**Compatibility note:**
- Users who depended on import-time logger setup still get logger setup on first settings access or first operation that needs settings.

**Files:**
- `src/qdmpy/__init__.py`
- `src/qdmpy/settings.py`
- docs/installation and docs/quickstart notes

---

### 4) Extension contract alignment (processors)

**Problem:** docs advertise protocol-based custom processors, while manager API type hints require `BaseProcessor`.

**Change (preferred):**
- Make manager accept `Processor` protocol at API boundary.
- Keep `BaseProcessor` as the recommended built-in base class.
- Preserve serialization support only for built-in/BaseProcessor processors.

**Alternative (rejected for now):** require inheritance from `BaseProcessor` for all custom processors. This is stricter but reduces extensibility and contradicts existing docs.

**Files:**
- `src/qdmpy/odmr/processors.py`
- docs/tutorial extension examples

---

### 4b) Extension contract alignment (field sources)

**Problem:** `.qdm` load path reconstructs field sources through `FieldSource(**data)`.
This bypasses discriminated-union behavior and can erase subtype semantics.

**Change:**
- Deserialize using `TypeAdapter(FieldSourceType)` (or equivalent union validation)
  so `kind='magnetic'` returns `MagneticSource`, etc.
- Preserve backward compatibility for older files missing optional subtype fields by
  keeping robust validation errors and clear messages.

**Files:**
- `src/qdmpy/io/qdm.py`
- `src/qdmpy/field_source.py` (if helper adapter is added)
- tests for typed round-trip behavior

---

### 5) Boundary cleanup for result containers

**Problem:** architecture intent says `QDMResult` is a pure container, but it still includes convenience save/load wrappers.

**Change:**
- Keep wrappers for backward compatibility in this release.
- Mark wrappers as convenience delegation in docs and clarify canonical I/O is `qdmpy.io.*`.
- Add deprecation pathway proposal for a future QEP if maintainers want strict purity.

**Files:**
- `src/qdmpy/result.py` (docstring and deprecation annotations if needed)
- docs/api and quickstart

---

### 6) Validation and hygiene touch-ups

Small correctness improvements bundled with architecture hardening:

- Validate frequency range count in `FitManager._validate_inputs()` against `data.sizes['freq_range']`.
- Remove or repurpose dead `Measurement._outliers` state.
- Align stale docs/examples in `odmr/__init__.py` with current API.
- Remove package-source artifacts (`__pycache__`, `.ipynb_checkpoints`) from tracked tree and enforce ignore rules.
- Fix CLI version lookup to use the distribution name `qdmpy-core`.
- Add explicit contract tests: folded constraint passthrough, normalization-default docs/runtime alignment,
  and import side-effect behavior.

---

## Implementation Plan

### Phase A (correctness first)
1. `.qdm` fit states round-trip.
2. Frequency-range validation hardening.
3. Folded-constraint unification and override propagation.

### Phase B (boundary cleanup)
4. Remove eager `get_settings()` import side effect.
5. Processor contract alignment.
6. Result container docs/deprecation notes.

### Phase C (docs and hygiene)
7. Update stale docs/examples.
8. Clean tracked artifacts and add guardrails.
9. CLI/package identity consistency fix.
10. Add contract tests for critical UX invariants.

---

## Backward Compatibility

- Public fitting and loading APIs remain unchanged.
- `.qdm` files from older versions remain loadable.
- `QDMResult.save/load` remain available in this QEP (no hard removal).
- Processor duck-typing remains supported; serialization remains best-effort for non-BaseProcessor custom types.

---

## Risks and Mitigations

1. **Behavioral drift in folded fits**
   - Mitigation: add regression tests comparing old vs new path on fixed synthetic fixtures.

2. **Logging expectations in notebooks/scripts**
   - Mitigation: document lazy setup and provide one-liner explicit initialization pattern.

3. **Custom processor serialization assumptions**
   - Mitigation: clearly document that pipeline round-trip serialization is guaranteed for built-in processors.

---

## Test Plan

- Persistence: save/load `.qdm` preserves parameters, `states`, B111 cache, images, and field sources.
- Fitting: folded constraint overrides are honored and exercised in unit tests.
- Validation: invalid frequency-shape combinations fail fast with `DataValidationError`.
- Import semantics: importing `qdmpy` does not create config directory until settings/operations require it.
- Processor extensibility: protocol-only custom processor can be added and executed by manager.
- Field source fidelity: `.qdm` round-trip preserves subtype (`MagneticSource`, `UpwardContinuedSource`).
- Documentation examples run against current API signatures.
- CLI version semantics: installed package version is shown consistently.

---

## Acceptance Criteria

- No known persistence loss for fit convergence state.
- Single canonical folded-constraint flow.
- No mandatory side effects on `import qdmpy`.
- Extension docs and typing contracts match runtime behavior.
- `.qdm` field source round-trip preserves discriminated subclasses.
- `qdmpy --version` uses package identity consistent with `pyproject.toml`.
- Contract-test suite guards critical user-facing defaults and argument semantics.
- CI passes with updated tests and docs.

---

## Open Questions

1. Should `QDMResult.save/load` be formally deprecated now, or deferred to a follow-up QEP?
2. Do we want strict failure for non-serializable custom processors in `pipeline_config`, or best-effort fallback only?
3. Should import-time logging configuration ever be opt-in via env var for notebook convenience?
