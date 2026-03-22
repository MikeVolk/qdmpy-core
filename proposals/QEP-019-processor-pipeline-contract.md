# QEP-019: Fix Processor Pipeline Contract Violations

**Status:** Implemented (QEP-032, 2026-02-20)
**Priority:** High
**Affects:** `odmr/processors.py`

## Problem

The processor pipeline has two contract violations that make it unreliable and
confusing.

### 1. FluorescenceCorrectionProcessor breaks the BaseProcessor interface

`BaseProcessor.process()` defines the contract:

```python
class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data: ODMRData) -> ODMRData: ...
```

But `FluorescenceCorrectionProcessor.process()` adds extra keyword arguments:

```python
def process(
    self,
    data: ODMRData,
    *,
    correction_factor: float | None = None,
    glob_fluorescence: float | None = None,
) -> ODMRData:
```

The `ODMRProcessorManager` calls `processor.process(data)` without any extra
arguments (processors.py:319), so these parameters are **never reachable**
through the pipeline. Users who pass a `FluorescenceCorrectionProcessor` to the
manager get the instance's `self.correction_factor` regardless of what they
intended.

This violates the Liskov Substitution Principle: a `FluorescenceCorrectionProcessor`
cannot be substituted wherever a `BaseProcessor` is expected without changing
behavior expectations.

### 2. OutlierProcessor threshold semantics are opaque

```python
def __init__(self, threshold: float = 0.001):
    self.threshold = threshold

def process(self, data):
    z_scores = np.abs((data.data - data_mean) / (data_std + 1e-10))
    mask = z_scores > (self.threshold * 3)  # Why * 3?
```

The `threshold` parameter is multiplied by 3 internally without documentation.
A user passing `threshold=3.0` (expecting a 3-sigma cutoff) actually gets a
9-sigma cutoff. The parameter name and default value (0.001) give no hint that
it is internally tripled.

### 3. Backward-compatibility alias serves no purpose

```python
visualize_fluorescence_correction = preview_fluorescence_correction
```

Per CLAUDE.md: "Avoid backwards-compatibility hacks like renaming unused _vars,
re-exporting types." This alias should be removed.

## Proposed Fix

### Fix 1: Align FluorescenceCorrectionProcessor with BaseProcessor

Remove the extra `process()` parameters. All configuration goes through
`__init__`:

```python
class FluorescenceCorrectionProcessor(BaseProcessor):
    def __init__(self, correction_factor: float = 0.2) -> None:
        self.correction_factor = correction_factor

    def process(self, data: ODMRData) -> ODMRData:
        # Uses self.correction_factor only — no extra kwargs
        ...
```

If users need to change the correction factor, they create a new processor
instance (immutable configuration pattern, consistent with BinningProcessor
and OutlierProcessor).

### Fix 2: Rename and document OutlierProcessor threshold

Rename the parameter to what it actually controls:

```python
class OutlierProcessor(BaseProcessor):
    def __init__(self, z_score_threshold: float = 0.003) -> None:
        """Mask pixels whose z-score exceeds z_score_threshold."""
        self.z_score_threshold = z_score_threshold

    def process(self, data: ODMRData) -> ODMRData:
        mask = z_scores > self.z_score_threshold  # Direct comparison, no * 3
```

This makes the parameter self-documenting and eliminates the hidden multiplier.

### Fix 3: Remove the alias

Delete `visualize_fluorescence_correction = preview_fluorescence_correction`.

## Validation

- Existing tests for processors should pass after renaming (update test params).
- Add test that all `BaseProcessor` subclasses have a `process(data)` signature
  compatible with `ODMRProcessorManager`.
- Add test that `ODMRProcessorManager.process()` passes a
  `FluorescenceCorrectionProcessor` through the full pipeline.

## Files to change

| File | Change |
|------|--------|
| `src/QDMpy/odmr/processors.py` | Fix process() signature, rename threshold, remove alias |
| `tests/odmr/test_processors.py` | Update tests for new parameter names |
