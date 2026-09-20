# QEP-ODMR-002 — Processor Protocol/ABC Unification

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-2)
**Module:** `odmr/processors.py`, `field_processing.py`
**Revised:** 2026-09-19 -- drift update; scope extended to field processors; decision set to a protocol contract with an optional base class

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

### Status update (2026-09-19)

Most of the motivation above has been addressed by later work (the
registry/config round-trip fix in `c4ec276` and the 2026-08-30 review fixes):

- There is no `ProcessorPipeline` Union any more. `ProcessorRegistry`
  resolves processors by their `type: Literal[...]` tag
  (`@ProcessorRegistry.register`), and
  `ODMRProcessorManager.from_config()` goes through it.
- `Processor` requires `process()` and `describe()`;
  `ODMRProcessorManager.add_processor` is typed `Processor`.
- `ProcessorRegistry.removed` records processor types that were removed
  (`OutlierProcessor`, 2026-09-19); `from_config` on one raises
  `ConfigurationError` with the reason.

Two gaps remain, and they are now the scope of this QEP:

1. **A duck-typed processor still cannot be saved and restored.**
   `ODMRProcessorManager.pipeline_config` emits `{"describe": ...}` for any
   processor that is not a `BaseProcessor`, which `from_config` cannot load.
   `ProcessorRegistry.register` is typed `type[BaseProcessor]` and reads
   `model_fields`, so it only accepts pydantic subclasses.
2. **Field processors have none of this.** `field_processing.py` has
   `BaseFieldProcessor` and `FieldProcessingPipeline`, but no protocol, no
   registry, no `type` tag and no `to_config`/`from_config`. A post-fit
   pipeline (`HotPixelFilter`, `QuadraticBackgroundSubtractor`,
   `UpwardContinuation`, `BlankSubtractor`) cannot be saved, and a custom
   field processor has no documented contract.

## GUI Integration Requirements

1. **Touchpoints.** `qdmpy_gui.app.pipeline_vm` and `widgets/load_dialog.py`
   call `ODMRProcessorManager.from_config`, `pipeline_config` and
   `ProcessorRegistry.removed`; `app/measurement_vm.py` and `app/workers.py`
   use `odmr.processor_manager.processors` / `add_processor`. The GUI has no
   field-processing UI today.
2. **Persisted data.** The `qdmpy_pipeline.json` sidecar format is unchanged
   for built-in processors (same `type` tag and fields), so no migration is
   needed.
3. **User-facing behaviour.** None for ODMR processors. A config naming an
   unknown type still raises `KeyError`, and a removed type still raises
   `ConfigurationError` (the GUI strips removed types before loading).
4. **Acceptance.** Load a folder -> edit the pipeline -> save the sidecar ->
   reopen -> the identical pipeline is restored. GUI tests pass without
   changes.
5. A future GUI field-processing panel can persist pipelines through
   `FieldProcessingPipeline.to_config()`; that panel is out of scope here.

## Proposed Changes

### Option A: Protocol-only (original recommendation, not chosen)

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
    NormalizationProcessor | BinningProcessor | FluorescenceCorrectionProcessor,
    Field(discriminator='processor_type'),
]
```

(Superseded: the type-tag registry already replaced the Union. See Option C.)

### Option B: ABC-only

Remove the `Processor` Protocol. Require all processors to inherit
`BaseProcessor`. Simpler but violates the Interface Segregation principle and
forces external users into Pydantic inheritance.

### Option C: protocol contract with an optional base class (decided 2026-09-19)

The protocol is the contract. `BaseProcessor` stays, but only as a
convenience that fills in part of it. The contract grows to cover
serialisation, so any conforming class can be saved and restored:

```python
@runtime_checkable
class Processor(Protocol):
    type: str                                  # registry key, also the config tag
    def process(self, data: ODMRData) -> ODMRData: ...
    def describe(self) -> str: ...
    def to_config(self) -> dict[str, Any]: ... # must include "type"
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self: ...
```

`BaseProcessor` (pydantic, `frozen=True, extra="forbid"`) provides
`describe`, `to_config` (`model_dump()`) and `from_config`
(`model_validate`), so the built-in processors and existing subclasses do not
change.

One generic registry serves both domains:

```python
class TypeRegistry[P]:
    def __init__(self, kind: str) -> None: ...
    def register[C: type[P]](self, cls: C) -> C: ...   # key = cls.type default
    def get(self, type_name: str) -> type[P]: ...      # honours `removed`
    def from_config(self, config: dict[str, Any]) -> P: ...
    removed: dict[str, str]

ProcessorRegistry = TypeRegistry[Processor]("ODMR processor")
FieldProcessorRegistry = TypeRegistry[FieldProcessor]("field processor")
```

`ProcessorRegistry.register` / `.get` / `.from_config` / `.removed` keep their
current spelling (the GUI uses `from_config` and `removed`). The registry key
is read from the class's `type` attribute rather than from pydantic
`model_fields`, so a non-pydantic class can register.

Field processors get the same shape: a `FieldProcessor` protocol,
`BaseFieldProcessor` as the optional base (it gains a `type` tag), and
`FieldProcessingPipeline.to_config()` / `.from_config()`.

`pipeline_config` stops emitting the unloadable `{"describe": ...}`. A
processor that does not implement `to_config` makes `pipeline_config` raise
`ConfigurationError` that names the processor.

### Recommendation

**Option C** (decided 2026-09-19). The original recommendation was
**Option A** (Protocol-first, dropping `BaseProcessor` as an unnecessary
intermediate layer). C keeps A's contract but not the removal:
`BaseProcessor` now carries real shared behaviour (`frozen`,
`extra="forbid"`, config round-trip), and removing it would break existing
subclasses for no gain.

## Migration

(Option C; the Option A migration steps are no longer planned.)

- Built-in ODMR processors: no change.
- Custom `BaseProcessor` subclasses: no change.
- Custom duck-typed processors: must add `type`, `to_config` and
  `from_config` to be saved. Until then they still run in a pipeline, but
  `pipeline_config` raises `ConfigurationError` instead of emitting
  `{"describe": ...}`.
- `BaseFieldProcessor` subclasses gain a `type: Literal[...]` field and
  `@FieldProcessorRegistry.register`. External subclasses without one still
  run but cannot be serialised (same error as above).

## Test Plan

- [ ] Verify custom class satisfying `Processor` protocol works in pipeline
- [ ] Verify `list_processors()` returns `describe()` output
- [ ] Verify serialization round-trips for all built-in processors
- [ ] Verify `isinstance(custom, Processor)` returns True
- [ ] Verify a duck-typed processor with `type`/`to_config`/`from_config`
      round-trips through `pipeline_config` -> `from_config`
- [ ] Verify `pipeline_config` raises `ConfigurationError` for a processor
      without `to_config`
- [ ] Verify every built-in field processor round-trips through
      `FieldProcessingPipeline.to_config()` / `.from_config()`
- [ ] Verify `removed` types raise `ConfigurationError` in both registries
- [ ] Verify the sidecar written before this change loads unchanged
