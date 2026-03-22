# QEP-042 — Fix Top-Level API Surface

**Status:** Implemented (2026-02-21)
**Created:** 2026-02-21

---

## Motivation

`QDMpy/__init__.py` currently exports:

```python
from QDMpy.magnetic import MagneticMap
from QDMpy.settings import NvSettings, get_settings, reset_settings
from QDMpy.utils import is_pygpufit_available
from QDMpy.field_processing import (
    BaseFieldProcessor, BlankSubtractor, HotPixelFilter,
    QuadraticBackgroundSubtractor, UpwardContinuation,
    FieldProcessingPipeline,
)
```

These are advanced/optional components. The things a new user needs first —
`Measurement`, `ODMR`, `MatlabLoader`, `BinningProcessor` — are not exported
and require knowing deep import paths:

```python
from QDMpy.odmr.io import MatlabLoader       # buried 2 levels
from QDMpy.odmr.manager import ODMR         # buried 2 levels
from QDMpy.odmr.data import ODMRData        # buried 2 levels
from QDMpy.odmr.processors import BinningProcessor  # buried 3 levels
from QDMpy.measurement import Measurement   # buried 1 level
from QDMpy.fitting import FitManager        # somewhat accessible
```

`from QDMpy import *` or relying on IDE autocomplete on `QDMpy.` produces
nothing useful for any of the three user types.

---

## Goals

1. All user-facing classes accessible via `from QDMpy import X`.
2. Submodule imports remain valid — nothing is moved.
3. Developer extension points (`Model`, `Processor`, `ModelRegistry`) exported
   so they are discoverable (full treatment in QEP-045).
4. `__all__` is explicit and complete.

---

## Design

### Proposed `__init__.py` exports

```python
# --- Entry points (User 1) ---
from QDMpy.measurement import Measurement
from QDMpy.result import QDMResult          # new in QEP-041

# --- Data loading ---
from QDMpy.odmr.io import MatlabLoader
from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.manager import ODMR

# --- Processing ---
from QDMpy.odmr.processors import (
    BinningProcessor,
    NormalizationProcessor,
    FluorescenceCorrectionProcessor,
    OutlierProcessor,
)

# --- Fitting ---
from QDMpy.fitting import FitManager, FitResult, ModelRegistry

# --- Extension base types (User 3, see QEP-045) ---
from QDMpy.fitting.models import Model
from QDMpy.odmr.processors import Processor   # Protocol

# --- Magnetic reconstruction ---
from QDMpy.magnetic import MagneticMap

# --- Settings ---
from QDMpy.settings import NvSettings, get_settings, reset_settings
from QDMpy.utils import is_pygpufit_available

# --- Field processing ---
from QDMpy.field_processing import (
    BaseFieldProcessor,
    BlankSubtractor,
    HotPixelFilter,
    QuadraticBackgroundSubtractor,
    UpwardContinuation,
    FieldProcessingPipeline,
)

__all__ = [
    # Entry points
    'Measurement', 'QDMResult',
    # Data loading
    'MatlabLoader', 'ODMRData', 'ODMR',
    # Processing
    'BinningProcessor', 'NormalizationProcessor',
    'FluorescenceCorrectionProcessor', 'OutlierProcessor',
    # Fitting
    'FitManager', 'FitResult', 'ModelRegistry',
    # Extension points
    'Model', 'Processor',
    # Magnetic
    'MagneticMap',
    # Settings
    'NvSettings', 'get_settings', 'reset_settings', 'is_pygpufit_available',
    # Field processing
    'BaseFieldProcessor', 'BlankSubtractor', 'HotPixelFilter',
    'QuadraticBackgroundSubtractor', 'UpwardContinuation', 'FieldProcessingPipeline',
]
```

### Grouping principle

Exported in decreasing order of user-frequency: first the things every user
needs, last the advanced/optional components. The import order doubles as
documentation of the recommended mental model.

---

## Alternatives Considered

### A. Keep deep imports, document them clearly
Rejected. Documentation decays; the import path *is* the API. Correct exports
are self-maintaining.

### B. Namespace subpackages (`QDMpy.odmr.ODMR` only)
Rejected. Python scientific libraries (numpy, scipy, xarray) consistently expose
their primary objects at the top level. Consistency with ecosystem norms matters.

---

## Files to Change

| File | Change |
|------|--------|
| `src/QDMpy/__init__.py` | Replace current exports with proposed set; add `__all__` |
| `tests/test_imports.py` | **New or extend** — smoke-test that each name in `__all__` is importable from `QDMpy` |
