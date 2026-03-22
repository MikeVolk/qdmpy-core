# QEP-CORE-002 — QDMResult Type Safety

**Status:** Implemented
**Created:** 2026-02-22
**Severity:** CRITICAL (C-2)
**Module:** `result.py`

---

## Motivation

`QDMResult.fit_result` is typed as `Any`:

```python
# result.py:45
fit_result: Any  # FitResult — typed as Any to avoid circular import at module level
```

The comment claims this avoids a circular import, but no circular import
exists: `FitResult` (in `fitting/result.py`) does not import `QDMResult`
(in `result.py`). The layering is strictly one-directional:

```
result.py (QDMResult) → fitting/result.py (FitResult) → constants.py
```

With `Any`, Pydantic performs no validation on construction:

```python
QDMResult(fit_result="garbage")  # passes construction
result.b111  # AttributeError at runtime
```

Every delegated property (`centers`, `linewidths`, `chi2`, `b111`, etc.)
crashes with an unhelpful `AttributeError` instead of a clear validation error
at construction time.

Similarly, `reconstructor: Any | None` has the same issue.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Proposed Changes

### 1. Use string forward reference

```python
from __future__ import annotations  # already present

class QDMResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fit_result: 'FitResult'
    nv_axis: tuple[float, float, float] | None = None
    reconstructor: 'FieldReconstructor | None' = None
```

With `from __future__ import annotations`, the string `'FitResult'` is not
evaluated at class definition time. The import is needed only at validation
time, which we handle via `model_rebuild()` or a deferred import in
`model_post_init`.

### 2. Add the actual import

Since there is no circular dependency, simply import `FitResult` at the top:

```python
from QDMpy.fitting.result import FitResult

class QDMResult(BaseModel):
    fit_result: FitResult
    ...
```

This is the simplest fix. If `FieldReconstructor` creates an import cycle,
use `TYPE_CHECKING` for that one only:

```python
from QDMpy.fitting.result import FitResult

if TYPE_CHECKING:
    from QDMpy.magnetic_map import FieldReconstructor

class QDMResult(BaseModel):
    fit_result: FitResult
    reconstructor: FieldReconstructor | None = None
```

### 3. Remove the `_magnetic_map_cache: Any` annotation

Replace with the proper type:

```python
_magnetic_map_cache: MagneticMap | None = PrivateAttr(default=None)
```

## Migration

- Any code passing non-FitResult objects to `QDMResult(fit_result=...)` will
  now fail at construction with a clear Pydantic `ValidationError`.
- This is the desired behavior — catching type errors early.
- No public API change for correct usage.

## Test Plan

- [ ] Verify `QDMResult(fit_result="garbage")` raises `ValidationError`
- [ ] Verify `QDMResult(fit_result=valid_fit_result)` succeeds
- [ ] Verify all delegated properties work correctly
- [ ] Verify mypy/pyright reports no type errors on QDMResult
