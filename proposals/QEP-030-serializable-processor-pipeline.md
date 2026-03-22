# QEP-030 — Serializable Processor Pipeline via Pydantic BaseProcessor

**Status:** Implemented (2026-02-20)
**Created:** 2026-02-18
**Supersedes:** QEP-019 (all three issues become natural consequences of this design)

---

## Motivation

The processing pipeline has no durable record of its configuration. After
`odmr.process_data()`, `processed_data.metadata` contains a partial, inconsistent
record of what was applied:

- `BinningProcessor` writes `bin_factor` ✓
- `NormalizationProcessor` writes `normalized: True` but **not** the method ✗
- `OutlierProcessor` writes `threshold` but the value is tripled internally ✗
- `FluorescenceCorrectionProcessor` writes the factor ✓

The live `ODMRProcessorManager.processors` list holds the full configuration, but
it is in-memory only. There is no way to persist, inspect, or replay a pipeline
across sessions. `list_processors()` returns only class names — no parameters.

Additionally, processor validation is scattered: `BinningProcessor` has a manual
`if bin_factor <= 0: raise DataValidationError(...)`, while other processors
accept any value without checking. The `FluorescenceCorrectionProcessor.process()`
signature violates the `BaseProcessor` contract (QEP-019, Issue 1).

---

## Goals

1. Every processor is fully serializable to a plain JSON-compatible dict.
2. A complete pipeline can be reconstructed from that dict — no code changes
   required, just data.
3. `processed_data.metadata['pipeline']` contains the canonical record of what
   was applied, in order, with all parameters.
4. Processor parameter validation is declared, not imperative.
5. `ODMRProcessorManager` gains `from_config()` for pipeline reconstruction.
6. All QEP-019 issues are resolved as natural side-effects.

---

## Design

### 3.1 `BaseProcessor` becomes a Pydantic `BaseModel`

```python
from abc import abstractmethod
from pydantic import BaseModel, ConfigDict

class BaseProcessor(BaseModel):
    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def process(self, data: ODMRData) -> ODMRData: ...
```

`frozen=True` enforces immutable configuration: all processor config lives in
`__init__` fields, never mutated after construction. This is already the intended
pattern (QEP-019 Fix 1) and now enforced by the model.

Pydantic v2 supports `@abstractmethod` on `BaseModel` subclasses directly — no
metaclass conflict.

### 3.2 Discriminator field on each concrete processor

Each concrete processor carries a `type` literal field used as the discriminator
for reconstruction:

```python
from typing import Literal
from pydantic import Field

class NormalizationProcessor(BaseProcessor):
    type: Literal['NormalizationProcessor'] = 'NormalizationProcessor'
    method: str = 'max'

    def process(self, data: ODMRData) -> ODMRData: ...


class BinningProcessor(BaseProcessor):
    type: Literal['BinningProcessor'] = 'BinningProcessor'
    bin_factor: int = Field(gt=0)

    def process(self, data: ODMRData) -> ODMRData: ...


class OutlierProcessor(BaseProcessor):
    type: Literal['OutlierProcessor'] = 'OutlierProcessor'
    z_score_threshold: float = Field(default=3.0, gt=0)
    # Renamed from 'threshold'; internal *3 multiplier removed (QEP-019 Fix 2)

    def process(self, data: ODMRData) -> ODMRData: ...


class FluorescenceCorrectionProcessor(BaseProcessor):
    type: Literal['FluorescenceCorrectionProcessor'] = 'FluorescenceCorrectionProcessor'
    correction_factor: float = Field(default=0.2, gt=0)

    def process(self, data: ODMRData) -> ODMRData:
        # Signature matches BaseProcessor — no extra kwargs (QEP-019 Fix 1)
        ...
```

`Field(gt=0)` replaces all manual validation; Pydantic raises `ValidationError`
with a clear message on construction.

### 3.3 `ProcessorSpec` discriminated union

```python
from typing import Annotated, Union
from pydantic import Field

ProcessorSpec = Annotated[
    Union[
        NormalizationProcessor,
        BinningProcessor,
        OutlierProcessor,
        FluorescenceCorrectionProcessor,
    ],
    Field(discriminator='type'),
]
```

Reconstruction from a raw dict becomes a one-liner:

```python
from pydantic import TypeAdapter

_adapter = TypeAdapter(ProcessorSpec)
processor = _adapter.validate_python({"type": "BinningProcessor", "bin_factor": 4})
```

### 3.4 Serialization: `BaseProcessor.to_config()`

A non-abstract helper on `BaseProcessor`:

```python
def to_config(self) -> dict[str, Any]:
    return self.model_dump()   # includes 'type' field
```

Produces e.g. `{"type": "BinningProcessor", "bin_factor": 4}`.

### 3.5 `ODMRProcessorManager` changes

**`process()` — write pipeline snapshot to metadata:**

```python
def process(self, data: ODMRData) -> ODMRData:
    pipeline_config = [p.to_config() for p in self.processors]
    for processor in self.processors:
        data = processor.process(data)
    metadata = data.metadata.copy()
    metadata['pipeline'] = pipeline_config
    from QDMpy.odmr.data import ODMRData as _ODMRData
    return _ODMRData(data=data.data, metadata=metadata)
```

The snapshot is taken *before* processing so it reflects configuration, not
output. It replaces the per-step ad-hoc metadata writes in each processor —
processors no longer write to metadata at all.

**New `from_config()` classmethod:**

```python
@classmethod
def from_config(cls, config: list[dict[str, Any]]) -> ODMRProcessorManager:
    """Reconstruct a pipeline from a serialized config list."""
    manager = cls()
    for step in config:
        manager.add_processor(_adapter.validate_python(step))
    return manager
```

**New `pipeline_config` property:**

```python
@property
def pipeline_config(self) -> list[dict[str, Any]]:
    return [p.to_config() for p in self.processors]
```

**`list_processors()` — keep but enhance:**

```python
def list_processors(self) -> list[str]:
    return [p.type for p in self.processors]
```

### 3.6 Processors no longer write to metadata

Currently every `process()` method manually copies and mutates `data.metadata`.
After this QEP, processors receive `ODMRData` and return a new `ODMRData` with
**only the data transformed** — metadata is not touched. The manager owns the
single metadata write (pipeline config). This eliminates the inconsistent
per-processor metadata records.

### 3.7 Typical usage — before and after

**Before:**
```python
manager.add_processor(BinningProcessor(bin_factor=4))
manager.add_processor(NormalizationProcessor())
odmr.process_data()

# Audit trail — incomplete, no method recorded
odmr.processed_data.metadata  # {'binned': True, 'bin_factor': 4, 'normalized': True}

# Reconstruction — not possible
```

**After:**
```python
manager.add_processor(BinningProcessor(bin_factor=4))
manager.add_processor(NormalizationProcessor())
odmr.process_data()

# Audit trail — complete
odmr.processed_data.metadata['pipeline']
# [
#   {"type": "BinningProcessor", "bin_factor": 4},
#   {"type": "NormalizationProcessor", "method": "max"},
# ]

# Reconstruction
manager2 = ODMRProcessorManager.from_config(odmr.processed_data.metadata['pipeline'])
```

---

## Removed

| Removed | Reason |
|---------|--------|
| Manual `if ... raise` in `BinningProcessor.__init__` | Replaced by `Field(gt=0)` |
| Per-processor metadata writes (`metadata["normalized"] = True`, etc.) | Manager owns the single pipeline snapshot |
| Extra kwargs on `FluorescenceCorrectionProcessor.process()` | Config is in fields (QEP-019 Fix 1) |
| `threshold` param + internal `* 3` in `OutlierProcessor` | Renamed `z_score_threshold`, applied directly (QEP-019 Fix 2) |
| `visualize_fluorescence_correction` alias | Dead re-export (QEP-019 Fix 3) |

---

## Alternatives Considered

### A. Abstract `to_config()` method only, processors stay plain classes

Simpler. But leaves validation scattered, does not enable `from_config()`
reconstruction without a manual registry, and misses the opportunity to make
the processor *itself* the validated config object.

### B. Separate config dataclass per processor (`BinningConfig`, etc.)

Doubles the class count with no added value. The processor IS its config — there
is no state beyond its Pydantic fields.

### C. Use `dataclasses` with `__post_init__` validation

No discriminated union support without extra work. Pydantic is already a
dependency (`ODMRData` is a Pydantic model), so there is no new dependency cost.

---

## Interactions

- **QEP-019:** Fully superseded. All three issues are resolved as natural
  consequences of the Pydantic model design.
- **QEP-007 (data validation):** Complementary. Processor field constraints
  add a second validation layer before data even reaches `ODMRData.validate_data_array`.

---

## Migration

Existing call sites constructing processors are unaffected — keyword arguments
map directly to Pydantic fields. The only breaking change is `OutlierProcessor`:
rename `threshold` → `z_score_threshold` and remove the implicit `* 3` factor
(callers using the default get equivalent behaviour with the new default of `3.0`).

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/odmr/processors.py` | `BaseProcessor(BaseModel, frozen)`; add `type` literals; field-based validation; remove per-processor metadata writes; add `ProcessorSpec` union + `_adapter`; update `ODMRProcessorManager.process()`, add `from_config()`, `pipeline_config` |
| `tests/odmr/test_processors.py` | Update `OutlierProcessor` param name; add round-trip serialization tests; add `from_config` reconstruction test |
