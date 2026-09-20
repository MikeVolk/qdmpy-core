# QEP-ODMR-003 — MatlabLoader Decomposition and Hardening

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-4, H-8, H-9)
**Module:** `odmr/io.py`

---

## Motivation

`MatlabLoader.load()` is a 90-line method with three `# noqa` suppressions
(`C901`, `PLR0912`, `PLR0915`) — the linter is telling us it's too complex.
It handles file discovery, MATLAB parsing (two backends), stack extraction,
frequency parsing, polarity stacking, spatial reshaping, coordinate building,
and xarray construction. Three specific issues:

1. **H-4: God Method.** Untestable as a unit — the only way to exercise the
   happy path is to provide real `.mat` files or mock at the `loadmat` level.

2. **H-8: Bare `except Exception`.** Line 96 catches *any* exception from
   `mat73.loadmat` and silently falls through to `scipy.io.loadmat`. This
   masks genuine errors (corrupt files, permission denied, OOM) and the scipy
   fallback uses pickle-capable loading by default.

3. **H-9: No dimension bounds.** `imgNumRows` and `imgNumCols` from `.mat`
   files are cast to `int` with no upper bound. A malformed file claiming
   `imgNumRows=1_000_000_000` causes an OOM crash at the `reshape` call.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Proposed Changes

### 1. Decompose `load()` into focused helpers

```python
class MatlabLoader(BaseLoader):
    def load(self) -> xr.DataArray:
        files = self._discover_files()
        per_file, rows, cols, frequencies = self._load_all_files(files)
        raw_data = self._stack_polarities(per_file)
        raw_data = self._reshape_spatial(raw_data, rows, cols)
        freq_ghz = self._build_freq_coord(frequencies, raw_data.shape[1])
        return self._to_xarray(raw_data, freq_ghz)

    def _discover_files(self) -> list[str]:
        """Find and sort run_*.mat files in data_folder."""
        ...

    def _load_mat_file(self, path: str) -> dict[str, Any]:
        """Load a single .mat file, trying mat73 then scipy."""
        ...

    def _load_all_files(self, files: list[str]) -> tuple[...]:
        """Load all files, extract stacks, dimensions, frequencies."""
        ...

    def _stack_polarities(self, per_file: list[NDArray]) -> NDArray:
        """Stack per-file data along polarity axis."""
        ...

    def _reshape_spatial(self, data: NDArray, rows: int, cols: int) -> NDArray:
        """Reshape flat pixels to (n_pol, n_frange, rows, cols, n_freq)."""
        ...

    def _build_freq_coord(self, freq: NDArray, n_frange: int) -> NDArray:
        """Build 2D freq_ghz coordinate from raw Hz frequencies."""
        ...

    def _to_xarray(self, data: NDArray, freq_ghz: NDArray) -> xr.DataArray:
        """Wrap numpy array in labelled xr.DataArray."""
        ...
```

Each helper is independently testable with synthetic data.

### 2. Narrow the except clause

```python
def _load_mat_file(self, path: str) -> dict[str, Any]:
    try:
        return mat73.loadmat(path)
    except (NotImplementedError, TypeError, ValueError) as exc:
        logger.debug(f'mat73 failed ({exc!r}), falling back to scipy')
        return loadmat(path)
```

`mat73.loadmat` raises `NotImplementedError` for v5 files — that's the only
expected failure mode. `TypeError`/`ValueError` cover edge cases. Genuine
`OSError`, `PermissionError`, `MemoryError` now propagate.

### 3. Add dimension bounds validation

```python
MAX_PIXELS_PER_AXIS = 50_000  # generous upper bound

def _validate_dimensions(self, rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0:
        raise DataValidationError(f'Invalid dimensions: {rows}x{cols}')
    if rows > MAX_PIXELS_PER_AXIS or cols > MAX_PIXELS_PER_AXIS:
        raise DataValidationError(
            f'Dimensions {rows}x{cols} exceed maximum {MAX_PIXELS_PER_AXIS}'
        )
```

Called in `_load_all_files` immediately after reading `imgNumRows`/`imgNumCols`.

## Migration

- No public API change — `MatlabLoader(folder).load()` returns the same
  `xr.DataArray`.
- Internal helpers are private (`_`-prefixed).
- The narrowed except clause may surface previously-swallowed errors for
  corrupt `.mat` files — this is intentional and correct.

## Test Plan

- [ ] Unit test each helper with synthetic numpy data (no real `.mat` files)
- [ ] Test `_load_mat_file` with mat73 failure → scipy fallback
- [ ] Test `_load_mat_file` with genuine `OSError` → propagates
- [ ] Test `_validate_dimensions` rejects 0, negative, and >50k values
- [ ] Integration test with real `.mat` files (gated on `pytest.mark.integration`)
- [ ] Remove all three `# noqa` suppressions from `load()`
