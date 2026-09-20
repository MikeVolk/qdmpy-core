# QEP-054 — Diamond Detection in Folded Fitting

**Status:** Draft
**Created:** 2026-03-09

---

## Motivation

Two related issues with `fit_folded_odmr`:

1. **Detection uses folded data**: when `model='auto'`, the peak detector runs
   on the folded spectrum. Folded spectra have a different shape from raw spectra
   (they are symmetric around the folding centre), which makes reliable dip
   counting harder. Detection should always use the **raw** (unfolded) data.

2. **`model_name` not forwarded in all paths**: `Measurement.fit_folded_odmr()`
   already accepts `model_name`, but it is not always correctly propagated when
   `FitManager.fit_folded()` internally falls back to the folded data for
   detection (see `manager.py` lines 619–630).

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Current Behaviour

`FitManager.fit_folded()` at lines 619–630 already contains logic to prefer
raw data over folded data for auto-detection:

```python
# Resolve auto model if needed — use raw (unfolded) data for reliable peak detection.
if raw_data is not None:
    detection_flat = raw_data.reshape(...)
else:
    detection_flat = data_5d.reshape(...)   # falls back to folded
self._resolve_auto_model(detection_flat)
```

However `Measurement.fit_folded_odmr()` calls:

```python
fit_result = fit_manager.fit_folded(resolved_folded, pixel_spacing=self.pixel_spacing)
```

`raw_data` is never passed — so the fallback path (folded data for detection)
is always taken, regardless of whether raw ODMR data is available on `self`.

---

## Design

### Pass raw data to `fit_folded`

`Measurement.fit_folded_odmr()` has access to `self.odmr.processed_data` (the
unfolded, processed `ODMRData`). Pass it through:

```python
fit_result = fit_manager.fit_folded(
    resolved_folded,
    pixel_spacing=self.pixel_spacing,
    raw_data=self.odmr.processed_data.data.values,   # unfolded 5D array
)
```

### Guarantee raw-only detection

In `FitManager.fit_folded()`, make the raw-data path the **only** path for
auto-detection and raise a clear error if neither raw data nor an explicit
`model_name` is provided:

```python
if self._model is None:
    if raw_data is None:
        msg = (
            "model='auto' requires raw (unfolded) ODMR data for detection. "
            "Pass raw_data= or set an explicit model_name."
        )
        raise DataValidationError(msg)
    self._resolve_auto_model(detection_flat)
```

### `model_name` propagation (already works, verify)

`Measurement.fit_folded_odmr(model_name=...)` already passes it to
`FitManager(model_name=model_name)`, which bypasses auto-detection entirely.
Verify with a test that explicit `model_name` skips detection even when raw
data is absent.

---

## Implementation Plan

1. Update `Measurement.fit_folded_odmr()` to pass `raw_data` to
   `fit_manager.fit_folded()`
2. Harden `FitManager.fit_folded()` to require either raw data or explicit
   model for auto mode; raise `DataValidationError` otherwise
3. Tests:
   - Unit: auto-detection uses raw data, not folded data
   - Unit: explicit `model_name` bypasses detection
   - Unit: missing raw data + `model='auto'` raises `DataValidationError`
4. Update CHANGELOG

---

## Acceptance Criteria

- `fit_folded_odmr(model_name='auto')` uses the unfolded processed data for
  model detection
- `fit_folded_odmr(model_name='ESR14N')` skips detection entirely
- No silent fallback to folded data for detection
