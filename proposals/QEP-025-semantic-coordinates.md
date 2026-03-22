# QEP-025 — Semantic Coordinates for ODMR Data and Fit Results

**Status:** Implemented
**Created:** 2026-02-18

---

## Motivation

The current data representation carries three sources of unnecessary confusion:

### 1. Opaque coordinate labels

`ODMRData` uses `pol_0`/`pol_1` and `frange_0`/`frange_1` as xarray coordinate values.
These tell the reader nothing about the physical meaning:

```python
data.sel(polarity='pol_0', freq_range='frange_1')   # what does this mean?
data.sel(polarity='neg',   freq_range='high')        # immediately obvious
```

Which file maps to which polarity is an implicit positional convention (first file
alphabetically = negative field) that lives nowhere in the data itself.

### 2. Ambiguous `delta_resonance` tensor

`_calc_delta_from_single_center` returns shape `(n_pol, 2, H, W)` where axis-1 is an
artificial ±1 sign dimension. This caused a critical B111 bug (fixed in 7b49a59) because
`_compute_b111` misread axis-1 as a polarity axis. The shape `(2, 2, H, W)` when `n_pol=2`
is ambiguous — two dimensions of size 2 with different meanings.

### 3. B111 results as bare ndarrays

`FitResult.b111` returns `(ndarray, ndarray)` — two spatially-dimensioned arrays with no
coordinate information. Downstream code must track which is remanent vs induced and what
the spatial axes mean out-of-band.

---

## Goals

1. Replace `pol_0`/`pol_1` with `neg`/`pos` coordinate labels throughout.
2. Replace `frange_0`/`frange_1` with `low`/`high` coordinate labels throughout.
3. Eliminate the artificial ±sign axis in `delta_resonance`; replace with a named
   `polarity_sign` or `component` coordinate that is self-describing.
4. Return B111 field results as an `xr.Dataset` with named variables and spatial coordinates.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### 2.1 Coordinate labels

**`ODMRData` / `MatlabLoader`** — change label generation:

```python
# Before
polarity_labels = [f'pol_{i}' for i in range(n_pol)]
frange_labels   = [f'frange_{i}' for i in range(n_frange)]

# After
POLARITY_LABELS = ['neg', 'pos']
FRANGE_LABELS   = ['low', 'high']
polarity_labels = POLARITY_LABELS[:n_pol]
frange_labels   = FRANGE_LABELS[:n_frange]
```

The physical assignment is already correct: files are loaded in alphabetical order
(`run_00000` → `neg`, `run_00001` → `pos`), and `frange_0` is always the low-frequency
branch (below ZFS). We just need the labels to reflect this.

Selection immediately becomes readable:

```python
odmr.data.sel(polarity='neg', freq_range='high')   # clear
odmr.data.sel(polarity='neg').mean('x')            # clear
```

### 2.2 `delta_resonance` — eliminate the ±sign axis

Instead of `(n_pol, 2, H, W)` with an opaque sign dimension, produce a `(n_pol, H, W)`
array with the polarity-correct sign already applied. This matches old QDMpy exactly and
removes the ambiguity that caused the B111 bug.

```python
# _calc_delta_from_single_center — before
d = np.array([-1, 1]).reshape(1, 2, 1, 1)
freq_diff = (resonance[:, 1] - resonance[:, 0]).reshape(n_pol, height, width)
return freq_diff[:, np.newaxis] / 2 / GAMMA_NV * 1e6 * d  # (n_pol, 2, H, W)

# After — apply sign per polarity, return (n_pol, H, W)
d = np.array([-1, 1]).reshape(n_pol, 1, 1)          # one sign per polarity
freq_diff = (resonance[:, 1] - resonance[:, 0]).reshape(n_pol, height, width)
return freq_diff / 2 / GAMMA_NV * 1e6 * d           # (n_pol, H, W)
```

Return as a named `xr.DataArray`:

```python
xr.DataArray(
    delta,
    dims=('polarity', 'y', 'x'),
    coords={'polarity': ['neg', 'pos']},
    attrs={'units': 'µT', 'description': 'signed dB per polarity'},
)
```

`_compute_b111` then unpacks naturally:

```python
neg_diff = delta_res.sel(polarity='neg').values   # (H, W)
pos_diff = delta_res.sel(polarity='pos').values   # (H, W)
b111_remanent = (neg_diff + pos_diff) / 2
b111_induced  = (neg_diff - pos_diff) / 2
```

No magic index arithmetic, no shape ambiguity.

### 2.3 B111 results as `xr.Dataset`

Replace the `(ndarray, ndarray)` tuple with a Dataset:

```python
@property
def b111(self) -> xr.Dataset:
    b_rem, b_ind = self._compute_b111()
    h, w = b_rem.shape
    return xr.Dataset(
        {
            'remanent': xr.DataArray(b_rem, dims=('y', 'x'), attrs={'units': 'µT'}),
            'induced':  xr.DataArray(b_ind, dims=('y', 'x'), attrs={'units': 'µT'}),
        }
    )
```

Access is then:

```python
result.b111['remanent']          # xr.DataArray
result.b111['remanent'].values   # ndarray fallback
```

`b111_remanent` and `b111_induced` convenience properties are kept as shims returning
`.values` for backward compatibility during the transition.

---

## Alternatives Considered

### A. Keep `pol_0`/`pol_1`, document convention more clearly
Rejected. Documentation rots; semantic coordinates don't. The labelling fix is low-risk
and high-value.

### B. Make polarity/frange user-configurable labels
Rejected. Overengineering — the physical meaning is fixed for all standard QDM
measurements. Adding configuration adds complexity with no benefit.

### C. Collapse both frequency ranges into one monotonic freq axis
Considered but deferred. The two frequency ranges are fitted independently and have
different roles (low/high branch). A single axis would require sparse coordinates and
complicate downstream fitting. May revisit in a dedicated QEP.

### D. Keep `delta_resonance` as `(n_pol, 2, H, W)` but enforce its invariant via tests
Rejected. The bug demonstrates that a shape that requires external documentation to
interpret safely is a design flaw, not a test coverage gap.

---

## Migration

| Location | Change |
|---|---|
| `odmr/io.py` — `MatlabLoader.load` | use `['neg','pos']`, `['low','high']` labels |
| `odmr/data.py` — `ODMRData.from_numpy` | same label change |
| `odmr/data.py` — `EXPECTED_DIMS` | no change (dim names stay the same) |
| `result.py` — `_calc_delta_from_single_center` | return `(n_pol, H, W)`, apply sign per pol |
| `result.py` — `_calc_delta_from_multi_centers` | same |
| `result.py` — `_compute_delta_resonance` | return named `xr.DataArray` |
| `result.py` — `_compute_b111` | use `.sel(polarity=...)` instead of index arithmetic |
| `result.py` — `b111` property | return `xr.Dataset` |
| `result.py` — `b111_remanent/induced` | keep as shims returning `.values` |
| Tests | update any hardcoded `pol_0`, `frange_1` coordinate assertions |

No changes to dimension *names* (`polarity`, `freq_range`) — only coordinate *values*.
The fitting pipeline (`fit.py`, `guess.py`) operates on raw numpy arrays and is unaffected.

---

## Impact Assessment

- **Risk:** Low. Coordinate label changes are isolated to loaders and `ODMRData`; the
  fitting pipeline works on `.values` throughout.
- **Breaking change:** Any user code that selects by `pol_0`/`frange_0` string will break.
  Since the API is still pre-1.0, this is acceptable.
- **Test changes:** Fixtures and assertions using string coordinate values need updating.
