# QEP-041 — QDMResult: Unified Result Container

**Status:** Implemented (2026-02-21)
**Created:** 2026-02-21

---

## Motivation

After fitting, the user has a `FitResult` containing fitted parameters and B111
field arrays. Converting to a full 3D magnetic map requires constructing a
`MagneticMap` manually:

```python
result = measurement.fit_odmr()
b111_da = xr.DataArray(
    result.b111_remanent,
    dims=('y', 'x'),
    attrs={'pixel_spacing': 4e-6},
)
mag_map = MagneticMap.from_b111(b111_da)
bz = mag_map.bz
```

This bridge is undocumented, requires knowing internals of both `FitResult` and
`MagneticMap`, and has a unit mismatch footgun (NDArray vs xr.DataArray with
attrs). The problem is not just ergonomics: the two objects conceptually form
one result from one measurement.

Placing `MagneticMap` inside `FitResult` would be a layering violation —
`fitting/` would import from the top-level magnetic reconstruction layer.
Placing it as a lazy property on `FitResult` hides the dependency.

The correct fix is an **outer container** returned by `Measurement`, which
already sits above both layers and owns `pixel_spacing` and NV geometry.

---

## Goals

1. `measurement.fit_odmr()` returns a `QDMResult` instead of a bare `FitResult`.
2. `QDMResult.fit_result` gives access to all existing `FitResult` properties.
3. `QDMResult.magnetic_map` lazily computes `MagneticMap` on first access.
4. `FitResult` and `MagneticMap` are unchanged — no layering violation.
5. Serialisation: `QDMResult.save()` / `QDMResult.load()` round-trips both
   objects.

---

## Design

### 4.1  `QDMResult`

```python
@dataclass(frozen=True)
class QDMResult:
    fit_result: FitResult
    pixel_spacing: float           # metres; needed for MagneticMap
    nv_axis: tuple[float, float, float] | None = None   # from settings if None

    # --- delegated convenience properties ---
    @property
    def b111_remanent(self) -> NDArray: ...   # delegates to fit_result
    @property
    def b111_induced(self) -> NDArray: ...
    @property
    def centers(self) -> NDArray: ...
    @property
    def chi2(self) -> NDArray: ...
    # … other high-frequency FitResult accessors

    # --- magnetic map (lazy) ---
    @cached_property
    def magnetic_map(self) -> MagneticMap: ...
    # builds b111 xr.DataArray with pixel_spacing attr, calls MagneticMap.from_b111()

    # --- IO ---
    def save(self, path: PathLike) -> None: ...
    @classmethod
    def load(cls, path: PathLike) -> QDMResult: ...
```

`QDMResult` lives in a new module `src/QDMpy/result.py` at the package root
(same level as `measurement.py`). It imports from `fitting/` (FitResult) and
top-level (MagneticMap) — both are downward dependencies, no circular import.

### 4.2  `Measurement.fit_odmr()` return type change

```python
# Before
def fit_odmr(self, ...) -> FitResult: ...

# After
def fit_odmr(self, ...) -> QDMResult: ...
```

`pixel_spacing` is already on `self`; `nv_axis` is read from settings if not
overridden.

### 4.3  Lazy `magnetic_map`

```python
@cached_property
def magnetic_map(self) -> MagneticMap:
    b111_da = xr.DataArray(
        self.fit_result.b111_remanent,
        dims=('y', 'x'),
        attrs={'pixel_spacing': self.pixel_spacing},
    )
    return MagneticMap.from_b111(b111_da, nv_axis=self.nv_axis)
```

`MagneticMap` reconstruction (Fourier inversion on 2k×2k arrays) is expensive;
users who only need B111 pay no cost. `@cached_property` works here because
`QDMResult` is a frozen dataclass — the underlying `fit_result` and
`pixel_spacing` never change.

> **Note**: `frozen=True` and `@cached_property` are compatible in Python ≥3.12
> via `__dict__` (dataclass frozen only blocks `__setattr__` on the instance,
> not descriptor `__set__`). Add `__hash__` manually if needed.

### 4.4  Delegation pattern

Delegate the most-used `FitResult` properties directly on `QDMResult` so users
rarely need `.fit_result.x`:

| `QDMResult.x` | delegates to |
|---------------|-------------|
| `b111_remanent` | `fit_result.b111_remanent` |
| `b111_induced` | `fit_result.b111_induced` |
| `b111` | `fit_result.b111` |
| `centers` | `fit_result.centers` |
| `linewidths` | `fit_result.linewidths` |
| `contrasts` | `fit_result.contrasts` |
| `chi2` | `fit_result.chi2` |
| `fit_states` | `fit_result.fit_states` |
| `get_parameter_map(name)` | `fit_result.get_parameter_map(name)` |
| `scan_dimensions` | `fit_result.scan_dimensions` |

### 4.5  User-facing API after this QEP

**User 1 — fit and be done:**
```python
result = measurement.fit_odmr()
result.b111_remanent          # NDArray, works as before
result.magnetic_map.bz        # free — no bridging code needed
```

**User 2 — exploratory:**
```python
result = measurement.fit_odmr()
result.b111.remanent.plot()               # xarray plot
result.get_parameter_map('center')        # NDArray (H, W)
result.magnetic_map.to_dataset().to_netcdf('output.nc')
```

**User 3 — custom pipeline (bypass Measurement):**
```python
fit_result = FitManager('ESR14N').fit(data, freq)
qdm_result = QDMResult(fit_result=fit_result, pixel_spacing=4e-6)
qdm_result.magnetic_map.bz
```

---

## Alternatives Considered

### A. Lazy property on `FitResult`
Rejected — downward dependency violation: `fitting/result.py` would import
`MagneticMap` from the top-level magnetic module.

### B. `MagneticMap.from_fit_result(result, pixel_spacing)` classmethod
Viable as a standalone convenience, but does not reduce boilerplate — the user
still constructs two objects and must remember to call `from_fit_result`. Kept
as an additional convenience method but not the primary solution.

### C. Return `(FitResult, MagneticMap)` tuple
Rejected. Named access (`result.magnetic_map`) is more readable and the tuple
pattern forces eager evaluation.

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/result.py` | **New file** — `QDMResult` dataclass |
| `src/QDMpy/measurement.py` | `fit_odmr()` return type `FitResult` → `QDMResult` |
| `src/QDMpy/__init__.py` | Export `QDMResult` |
| `tests/test_result.py` | **New file** — tests for `QDMResult` delegation, lazy `magnetic_map`, save/load |
| `tests/test_measurement.py` | Update assertions for new return type |
