# QEP-052 — Per-Frange Frequency Cutoff for Fitting

**Status:** Implemented
**Created:** 2026-03-09

---

## Motivation

The QDM instrument sweeps a low-frequency range (frange_0, below ZFS) and a
high-frequency range (frange_1, above ZFS). These ranges are deliberately
positioned so that the inner boundary sits at ZFS — the central ODMR peak that
is not of interest for field sensing.

Occasionally, strong applied fields or misaligned sweep ranges cause the inner
spectrum (clustered near ZFS) to bleed into the measurement window. When this
happens, `guess_model` and the fitter pick up spurious dips near the inner edge,
producing fit artefacts.

Currently there is no way to restrict the frequency band used for fitting.
Adding a per-frange cutoff (lower and upper frequency bounds) would let the user
exclude contaminated edge regions from the fit without changing the raw data.

---

## GUI Integration Requirements

1. **Core API contracts used by `qdmpy-gui`:**
   - `Measurement.fit_odmr(..., freq_cutoff=...)`
   - `Measurement.fit_folded_odmr(..., freq_cutoff=...)`
   - No changes to `QDMResult`/`FitResult` output field names or array shapes.
2. **GUI settings/session migration:**
   - No required migration. `freq_cutoff` is optional and defaults to `None`.
   - Existing saved sessions/configs without this key remain valid and unchanged.
3. **Expected GUI user behavior:**
   - Successful runs behave as before, but fit uses only the selected frequency window per range.
   - Invalid cutoff specifications raise `DataValidationError` with actionable messages.
   - No new progress states are required.
4. **GUI acceptance checks:**
   - Load dataset -> run fit with and without `freq_cutoff` -> verify output maps render.
   - Save/reload session with `freq_cutoff` present -> rerun fit -> verify identical behavior.
   - Submit invalid cutoff from GUI controls -> verify clear error is shown without crash.
5. **Impact rationale:**
   - Low impact to GUI integration because this is an additive fit-time parameter only.
   - Smoke check: existing GUI fit flow with `freq_cutoff=None` remains unchanged.

## Design

### New parameter: `freq_cutoff`

Add optional `freq_cutoff` to `FitManager` (and exposed via `Measurement.fit_odmr`
and `Measurement.fit_folded_odmr`):

```python
FitManager(
    model_name="auto",
    freq_cutoff={
        "low":  {"min": None, "max": 2.86},   # GHz — exclude near-ZFS edge
        "high": {"min": 2.88, "max": None},   # GHz — exclude near-ZFS edge
    }
)
```

Both `min` and `max` are optional per frange; `None` means no cutoff on that side.
Specifying only one frange leaves the other unrestricted.

Decision notes:

- Keep this as a plain dictionary contract (no `pydantic.BaseModel` for this QEP).
- Public schema is limited to `low/high` and `min/max`.
- No `inner/outer` aliases in this QEP to keep scope tight and avoid duplicate semantics.

### Where the cut is applied

Frequency masking is applied **before** fitting, immediately after the data is
flattened. The cut slices both the frequency axis and the corresponding data
array for that frange. The guesser and the gpufit call both see the reduced
frequency window.

Importantly the cut is applied **per frange separately** — frange_0 and frange_1
can have independent cutoffs. Polarity separation is not planned for this QEP
(both polarities share the same frange cutoffs).

For folded fitting (`n_frange=1`), the cutoff is optional and uses a single-range
mapping: only `low` is accepted as the active range key (`high` is rejected).

### Default behaviour

`freq_cutoff=None` — full frequency range, identical to current behaviour.
No breaking change.

---

## Implementation Plan

1. Add `freq_cutoff` parameter to `FitManager.__init__` (stored as
   `self._freq_cutoff: dict | None`).
2. Add explicit cutoff normalization/validation in `FitManager`:
   - allowed top-level keys: `low`, `high`
   - allowed inner keys: `min`, `max`
   - validate ordering (`min <= max`) and numeric/None types
   - validate enough points remain after mask (minimum 10 frequency points)
3. Apply cutoff masking in `fit()` per frange before guessing and GPU fitting.
4. Expose `freq_cutoff` in `Measurement.fit_odmr()` and
   `Measurement.fit_folded_odmr()` and pass through to `FitManager`.
5. Ensure `Measurement.refit_outliers()` uses the same cutoff behavior so refit
   and initial fit are consistent.
6. Tests:
   - Unit: cutoff masks correct frequency indices for each frange
   - Unit: `None` cutoff is a no-op
   - Unit: single-sided cutoff (min only, max only)
   - Unit: validation errors for unknown keys / invalid bounds
   - Unit: post-mask minimum frequency count enforcement
   - Integration: fitting with cutoff produces valid `FitResult`
7. Update CHANGELOG

---

## Alternatives Considered

- **Apply cutoff in processors**: rejected — processors transform data, not
  fitting metadata; frequency masking belongs in the fitting layer.
- **Per-polarity cutoffs**: deferred to a future QEP; adds complexity for a
  rare use case.
- **`BaseModel` cutoff config object**: rejected for this QEP — a plain dict is
  sufficient for a fit-time optional parameter and matches existing call-site
  style (`constraints`).
