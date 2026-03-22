# QEP-053 — Diamond Detection Regression (MIL Data)

**Status:** Draft
**Created:** 2026-03-09

---

## Motivation

The automatic diamond type detection (`model='auto'`) works correctly on
reference data (FOV18x) but misidentifies the model on MIL-series data.
This is a regression — MIL data was correctly identified in a previous
implementation.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Background

`guess_model()` in `fitting/guess.py` uses `guess_n_peaks()`:

1. Compute the **median spectrum** across pixels for each `(pol, frange)` slice
2. Detect dips via `scipy.signal.find_peaks(-spectrum, prominence=prominence)`
   where `prominence = _relative_prominence(spectrum)`
3. Take the **majority vote** across all `(pol, frange)` combinations

For 14N (ESR14N) the expected result is 3 dips per frange. For MIL data the
detector is returning an incorrect count.

---

## Known Risk Factors

- **Relative prominence threshold**: `_relative_prominence` is computed from
  the spectrum amplitude. If MIL spectra have lower contrast (shallower dips),
  the threshold may be too high, causing dips to be missed or extra noise peaks
  to be detected.
- **Spatial median**: the median is taken across all pixels. If MIL data has
  many blank or masked pixels (e.g., diamond edge, substrate) the median
  spectrum may be dominated by background rather than signal.
- **Binning interaction**: MIL data may be loaded at different bin factors;
  higher binning smooths spectra and can reduce peak sharpness.
- **Normalization order**: if the detection runs on non-normalized or
  differently normalized data, the prominence calculation changes.

---

## Investigation Plan

1. Load MIL data and inspect the median spectrum per `(pol, frange)` slice
2. Plot detected peaks with the prominence threshold overlaid
3. Compare `_relative_prominence` values for FOV18x vs MIL data
4. Check whether masked/blank pixels are included in the median

---

## Implementation Plan

1. Add `plot_model_detection(odmr_data)` diagnostic plot (already referenced in
   the warning log message but not yet exposed) — shows median spectra per slice
   with detected peaks marked
2. Fix the root cause identified in investigation (likely threshold or masking)
3. Add regression tests using MIL reference data that assert correct model
   detection
4. Update CHANGELOG

---

## Acceptance Criteria

- `model='auto'` correctly identifies ESR14N on MIL-series data
- `plot_model_detection()` is a callable diagnostic available to users
- No regression on FOV18x reference data
