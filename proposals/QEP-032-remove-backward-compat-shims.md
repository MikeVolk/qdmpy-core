# QEP-032: Remove Backward-Compatibility Shims and Clean Up Tech Debt

**Status:** Implemented
**Priority:** Medium
**Affects:** Multiple modules

## Context

QEP-023 (Feb 2026) reorganized fitting code into a `fitting/` subpackage and renamed
`odmr/odmr.py` → `odmr/manager.py`. To avoid breaking existing imports, 7
backward-compatibility shim files were left in place. Additionally, class-level
aliases (`parameters_unique`, `model_params_unique`) and exception aliases
(`CantImportError`, `WrongFileNumberError`) were preserved.

Two existing draft proposals (QEP-019, QEP-021) identify additional tech debt
that was never implemented.

This QEP removes all backward-compatibility shims and cleans up the dead code
identified in QEP-019 and QEP-021. All three are marked as implemented.

**Out of scope:** `FitResult` pickle-based serialization refactor (tracked separately).

## Changes

### Shim Files Removed

| File | Replaced by |
|------|-------------|
| `src/QDMpy/fit.py` | `QDMpy.fitting.manager` |
| `src/QDMpy/result.py` | `QDMpy.fitting.result` |
| `src/QDMpy/models.py` | `QDMpy.fitting.models` |
| `src/QDMpy/guess.py` | `QDMpy.fitting.guess` |
| `src/QDMpy/io.py` | `QDMpy.measurement` |
| `src/QDMpy/odmr/odmr.py` | `QDMpy.odmr.manager` |
| `src/QDMpy/odmr/validation.py` | `QDMpy.odmr.data` |

### Property Aliases Removed

- `Model.parameters_unique` — use `Model.parameter_names`
- `FitManager.model_params_unique` — use `FitManager.parameter_names`

### Exception Aliases Removed (QEP-021 §5)

- `CantImportError` — use `DependencyError`
- `WrongFileNumberError` — use `DataValidationError`

### ModelRegistry Fixed (QEP-021 §6, §7)

- `ModelRegistry.register()` now uses `model_cls.name` (ClassVar) instead of
  creating a throwaway instance
- Removed `ModelRegistry._initialize_constraints()` (duplicated `ConstraintManager.__init__`)
- Added `name: ClassVar[str]` to `ESR14N`, `ESR15N`, `ESRSINGLE`

### CLI Fixed (QEP-021 §1, §2)

- `models_command_handler` now prints model information to stdout
- Removed `process` and `info` subcommands (both raised `NotImplementedError`)
- Removed `__main__` block from `measurement.py` (hardcoded developer path)
- Removed `self._B111 = None` unused attribute from `Measurement.__init__`

### Processor Pipeline Fixed (QEP-019)

- `FluorescenceCorrectionProcessor.process()` signature now matches `BaseProcessor`
  (no extra kwargs — configure via `__init__` instead)
- `OutlierProcessor.threshold` renamed to `z_score_threshold`; default updated
  from `0.001` to `0.003` (was `0.001 * 3`); hidden `* 3` multiplier removed
- Removed `visualize_fluorescence_correction = preview_fluorescence_correction` alias

## Migration Guide

| Old import | New import |
|------------|------------|
| `from QDMpy.fit import FitManager` | `from QDMpy.fitting.manager import FitManager` |
| `from QDMpy.models import ESR14N` | `from QDMpy.fitting.models import ESR14N` |
| `from QDMpy.result import FitResult` | `from QDMpy.fitting.result import FitResult` |
| `from QDMpy.guess import cumsum_center` | `from QDMpy.fitting.guess import cumsum_center` |
| `from QDMpy.io import get_image` | `from QDMpy.measurement import get_image` |
| `from QDMpy.odmr.odmr import ODMR` | `from QDMpy.odmr.manager import ODMR` |
| `from QDMpy.odmr.validation import validate_frequencies` | `from QDMpy.odmr.data import validate_frequencies` |
| `model.parameters_unique` | `model.parameter_names` |
| `fit_manager.model_params_unique` | `fit_manager.parameter_names` |
| `OutlierProcessor(threshold=0.003)` | `OutlierProcessor(z_score_threshold=0.003)` |
| `processor.process(data, correction_factor=0.5)` | `FluorescenceCorrectionProcessor(0.5).process(data)` |

## Alternatives Considered

- **Keep shims indefinitely**: rejected — shims add confusion, maintenance burden,
  and obscure the canonical import paths.
- **Deprecation warnings before removal**: would require another release cycle;
  since the codebase is still pre-release, clean removal is preferable.
