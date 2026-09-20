# QEP-056-IO - Serialization Fidelity and Round-Trip Guarantees

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | QEP-008, QEP-050, QEP-051-ARCH |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-03-09 |

---

## Motivation

The `.qdm` format is now central to reproducibility, but fidelity guarantees are
still implicit and partially tested. Users expect save/load to preserve analysis
state and typed domain objects without surprises.

Key risks today:

1. Fit diagnostics can be partially persisted.
2. Typed field-source subclasses can be degraded on load.
3. Round-trip invariants are not specified as a strict contract.

---

## Goals

- Define a strict round-trip contract for `.qdm`.
- Guarantee preservation of fit diagnostics (`states`, `chi2`, metadata).
- Guarantee discriminated field-source round-trip fidelity.
- Keep backward compatibility for older `.qdm` files.

## Non-goals

- No `.qdm` major-version bump in this QEP.
- No change to core B111 physics computations.
- No mandatory inclusion of optional heavy datasets (Bxyz).

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Round-Trip Contract

For `result2 = load_qdm(save_qdm(result1))`, the following must hold:

1. `model_name`, `scan_dimensions`, `pixel_spacing` equal.
2. `parameters` preserve all stored keys and values within format precision.
3. `fit_states` is restored and available via `result2.fit_states`.
4. `field_sources` preserve subclass identity by `kind` discriminator.
5. `light_image` / `laser_image` preserve presence and shape.
6. `metadata` round-trips losslessly as JSON-compatible data.

---

## Design

### 1) Typed field source deserialization

- Replace base-class reconstruction with discriminated union validation
  (`FieldSourceType` adapter).
- Preserve fallback behavior for legacy generic entries.

### 2) Fit diagnostics fidelity

- Require restoration of `fit_states` from dedicated dataset when present.
- Keep tolerant loading when absent in older files.

### 3) Contract-focused tests

- Add fixture matrix: no images, with images, generic source, magnetic source,
  upward-continued source, with/without fit states.
- Test both new files and representative older fixtures.

---

## Files to Change

| File | Change |
|------|--------|
| `src/qdmpy/io/qdm.py` | Enforce typed/source and diagnostics round-trip behavior |
| `tests/test_io_qdm.py` | Expand with explicit contract matrix assertions |
| `reference_data/` | Optional legacy `.qdm` fixtures for compatibility tests |

---

## Risks and Mitigations

1. **Legacy file variability**
   - Mitigation: tolerant loader path with explicit warnings.
2. **Precision mismatch confusion**
   - Mitigation: define tolerance policy in tests/docs (`float32` datasets).

---

## Acceptance Criteria

- Contract checklist is documented and tested.
- No known data loss for covered fields.
- Legacy `.qdm` major-compatible files remain loadable.
