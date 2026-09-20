# QEP-ODMR-001 — ODMR Manager Immutability

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-1)
**Module:** `odmr/manager.py`

---

## Motivation

`ODMR` holds mutable state with a redundant `is_processed` bool that can
drift out of sync with `_processed_data`:

```python
# odmr/manager.py:29-38
self._raw_data = odmr_data
self._processed_data: ODMRData | None = None
self.is_processed = False              # redundant — derivable from _processed_data
self.processor_manager = ODMRProcessorManager()
```

Every method that changes state (`load_data`, `load_xarray`, `reset`,
`process_data`) must remember to update *both* `_processed_data` and
`is_processed`. Tests already bypass this by setting `odmr.is_processed = True`
directly (see `tests/odmr/test_odmr.py:77`), proving the invariant is weak.

Additionally, `load_data` / `load_xarray` / `reset` all return `self` for
chaining — a pattern that encourages in-place mutation rather than producing
new state.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Proposed Changes

### 1. Remove `is_processed` flag

Replace with a property derived from `_processed_data`:

```python
@property
def is_processed(self) -> bool:
    """Whether processed data is available."""
    return self._processed_data is not None
```

This eliminates the sync bug entirely.

### 2. Make `_raw_data` read-only after construction

Once set in `__init__`, raw data should not change. If the user wants
different raw data, construct a new `ODMR` instance. Remove `load_data` and
`load_xarray` as public mutators — they exist because the old API allowed
constructing an empty `ODMR()` and loading later.

```python
class ODMR:
    def __init__(self, odmr_data: ODMRData) -> None:
        self._raw_data: Final[ODMRData] = odmr_data
        self._processed_data: ODMRData | None = None
        self.processor_manager = ODMRProcessorManager()
```

If backward compatibility is needed, keep `load_data`/`load_xarray` but emit
a deprecation warning.

### 3. `process_data()` returns new `ODMR` instead of mutating

```python
def process_data(self) -> ODMR:
    """Apply processing pipeline and return a new ODMR with processed data."""
    processed = self.processor_manager.process(self._raw_data)
    new = ODMR(self._raw_data)
    new._processed_data = processed
    new.processor_manager = self.processor_manager
    return new
```

Or, more pragmatically, keep the mutation but remove `is_processed`:

```python
def process_data(self) -> Self:
    """Apply processing pipeline. Sets _processed_data."""
    self._processed_data = self.processor_manager.process(self._raw_data)
    return self
```

### 4. `reset()` clears only processed data

```python
def reset(self) -> Self:
    self._processed_data = None
    return self
```

No `is_processed = False` needed — the property handles it.

## Alternatives Considered

- **Full immutability (return new ODMR from process_data):** Cleaner but
  breaks existing `meas.odmr.process_data()` patterns in `Measurement`.
  Defer to a future QEP if desired.
- **Make ODMR a Pydantic model:** Overkill — `ODMR` is a lifecycle manager,
  not a data container.

## Migration

- Remove all direct writes to `odmr.is_processed` in tests.
- `Measurement` code that checks `self.odmr.is_processed` works unchanged
  (now a property).
- If `load_data`/`load_xarray` are deprecated, callers must construct new
  `ODMR(ODMRData.from_numpy(...))` instead.

## Test Plan

- [ ] Verify `is_processed` is True after `process_data()`, False after `reset()`
- [ ] Verify `is_processed` cannot be set directly (AttributeError)
- [ ] Verify `_raw_data` cannot be reassigned after construction
- [ ] Verify existing `Measurement` integration still works
