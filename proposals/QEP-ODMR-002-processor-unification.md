# QEP-ODMR-002 — Processor Protocol/ABC Unification

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-2)
**Module:** `odmr/processors.py`

---

## Motivation

`odmr/processors.py` defines both a `Processor` Protocol and a
`BaseProcessor` ABC. The docstring on `Processor` says "Processors do not need
to inherit from any base class — structural subtyping (duck typing) is used."
But `ODMRProcessorManager.list_processors()` calls `type(p).__name__`, and the
serialization/deserialization pipeline (`ProcessorPipeline` type adapter) only
recognises `Union[NormalizationProcessor, BinningProcessor, ...]` — a
hard-coded union of `BaseProcessor` subclasses.

A user who implements the `Processor` protocol without inheriting
`BaseProcessor` will:
1. Pass `isinstance(p, Processor)` checks (runtime_checkable).
2. Work in `process()` calls.
3. **Fail** `ProcessorPipeline` deserialization (not in the Union).
4. **Lack** `to_config()` / `describe()` defaults from `BaseProcessor`.

The two abstractions serve the same purpose and confuse the extension story.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Proposed Changes

### Option A: Protocol-only (recommended)

Remove `BaseProcessor` entirely. Make the built-in processors plain
`BaseModel`s that happen to satisfy the `Processor` protocol:

```python
@runtime_checkable
class Processor(Protocol):
    def process(self, data: ODMRData) -> ODMRData: ...
    def describe(self) -> str: ...

class NormalizationProcessor(BaseModel):
    model_config = ConfigDict(frozen=True)
    method: Literal['max', 'min', 'mean'] = 'max'

    def process(self, data: ODMRData) -> ODMRData:
        ...

    def describe(self) -> str:
        return f'NormalizationProcessor(method={self.method!r})'
```

`ODMRProcessorManager` stores `list[Processor]`:

```python
class ODMRProcessorManager:
    def __init__(self) -> None:
        self._processors: list[Processor] = []

    def add_processor(self, p: Processor) -> None:
        self._processors.append(p)

    def list_processors(self) -> list[str]:
        return [p.describe() for p in self._processors]
```

Serialization uses a discriminated union with a `type` field for built-in
processors, and a catch-all for custom ones:

```python
BuiltinProcessor = Annotated[
    NormalizationProcessor | BinningProcessor | OutlierProcessor | ...,
    Field(discriminator='processor_type'),
]
```

### Option B: ABC-only

Remove the `Processor` Protocol. Require all processors to inherit
`BaseProcessor`. Simpler but violates the Interface Segregation principle and
forces external users into Pydantic inheritance.

### Recommendation

**Option A** — Protocol-first aligns with the documented extension story and
Python typing best practices. `BaseProcessor` is an unnecessary intermediate
layer.

## Migration

- Built-in processors drop `BaseProcessor` inheritance, add protocol methods
  directly.
- `to_config()` moves to each processor (one-liner: `self.model_dump()`).
- Custom processors that inherit `BaseProcessor` still work (they satisfy the
  protocol). Emit deprecation warning on `BaseProcessor` for one release.
- `ProcessorPipeline` TypeAdapter updates to the new discriminated union.

## Test Plan

- [ ] Verify custom class satisfying `Processor` protocol works in pipeline
- [ ] Verify `list_processors()` returns `describe()` output
- [ ] Verify serialization round-trips for all built-in processors
- [ ] Verify `isinstance(custom, Processor)` returns True
- [ ] Verify `BaseProcessor` deprecation warning (if kept temporarily)
