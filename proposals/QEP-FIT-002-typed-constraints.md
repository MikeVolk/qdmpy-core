# QEP-FIT-002 — Typed Constraint Storage

**Status:** Implemented
**Created:** 2026-02-22
**Severity:** CRITICAL (C-4)
**Module:** `fitting/constraints.py`

---

## Motivation

`ConstraintManager` stores constraints as `dict[str, list[Any]]` with magic
positional indices: `[0]=vmin`, `[1]=vmax`, `[2]=type`, `[3]=units`. Access
sites use `current[0]`, `current[1]`, `current[2]` — fragile and opaque.
Adding a new constraint attribute (e.g. step size, initial scale) requires
updating every access site with no compile-time safety.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Current Code

```python
# constraints.py:35-51
self._constraints: dict[str, list[Any]] = {}
...
self._constraints[param] = [
    getattr(settings, f"{base_param}_min"),   # [0]
    getattr(settings, f"{base_param}_max"),   # [1]
    getattr(settings, f"{base_param}_type"),  # [2]
    units[param],                             # [3]
]

# constraints.py:71-80 — magic index access
current = self._constraints[param]
if vmin is not None:
    current[0] = vmin
if vmax is not None:
    current[1] = vmax
```

## Proposed Changes

### 1. Define `ParameterConstraint` frozen dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ParameterConstraint:
    """Immutable constraint specification for a single model parameter."""
    vmin: float
    vmax: float
    constraint_type: str  # 'FREE', 'LOWER', 'UPPER', 'LOWER_UPPER'
    units: str            # 'GHz' or 'a.u.'

    def __post_init__(self) -> None:
        if self.constraint_type not in CONSTRAINT_TYPES:
            msg = f"Invalid constraint type: {self.constraint_type}"
            raise ParameterError(msg)
        if self.constraint_type in ('LOWER', 'LOWER_UPPER') and self.vmin > self.vmax:
            msg = f"vmin ({self.vmin}) > vmax ({self.vmax})"
            raise ParameterError(msg)
```

### 2. Replace `dict[str, list[Any]]` with `dict[str, ParameterConstraint]`

```python
class ConstraintManager:
    def __init__(self, model: Model, settings: ModelConstraintsSettings) -> None:
        self._constraints: dict[str, ParameterConstraint] = {}
        self._model = model
        self._initialize_constraints(settings)

    def _initialize_constraints(self, settings: ModelConstraintsSettings) -> None:
        units = self._model.units
        for param in self._model.parameter_names:
            base = self._model.parameter_types[param]
            self._constraints[param] = ParameterConstraint(
                vmin=getattr(settings, f'{base}_min'),
                vmax=getattr(settings, f'{base}_max'),
                constraint_type=getattr(settings, f'{base}_type'),
                units=units[param],
            )
```

### 3. Update `set_constraint` to return new immutable constraint

```python
def set_constraint(self, param: str, **kwargs: Any) -> None:
    if param not in self._constraints:
        raise ParameterError(f"Unknown parameter: {param}")
    from dataclasses import replace
    current = self._constraints[param]
    updates = {k: v for k, v in kwargs.items() if v is not None}
    self._constraints[param] = replace(current, **updates)
```

### 4. Update `to_array` and `get_constraint_types`

```python
def to_array(self, n_pixel: int, parameter_names: list[str]) -> NDArray:
    bounds = []
    for param in parameter_names:
        c = self._constraints[param]
        bounds.extend((c.vmin, c.vmax))
    return np.tile(bounds, (n_pixel, 1))

def get_constraint_types(self, parameter_names: list[str]) -> NDArray:
    return np.array(
        [CONSTRAINT_TYPES.index(self._constraints[p].constraint_type)
         for p in parameter_names],
        dtype=np.int32,
    )
```

## Migration

- `get_constraints()` return type changes from `dict[str, list[Any]]` to
  `dict[str, ParameterConstraint]`. Callers accessing `[0]`, `[1]`, `[2]`
  must use `.vmin`, `.vmax`, `.constraint_type`.
- `FitManager.constraints` property return type changes accordingly.
- No public API change for `set_constraints()` — same kwargs.

## Test Plan

- [ ] Verify `ParameterConstraint` rejects invalid constraint_type
- [ ] Verify `ParameterConstraint` rejects vmin > vmax for bounded types
- [ ] Verify `set_constraint` produces new immutable instance
- [ ] Verify `to_array` output matches old behavior exactly
- [ ] Verify `get_constraint_types` output matches old behavior exactly
