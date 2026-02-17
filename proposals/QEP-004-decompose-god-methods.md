# QEP-004: Decompose God-Methods

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | L |
| **Depends on** | QEP-003 (QEP-001 superseded by QEP-011) |
| **Blocks** | QEP-008 |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |

## QEP-011 Impact

QEP-011 reduced `FitManager` from 703 to ~462 lines by leveraging xarray's named
dimensions for cleaner array manipulation. However, the structural decomposition
(extracting `ParameterGuesser`, splitting `Measurement.fit_odmr()`) is still needed.
`Measurement.fit_odmr()` remains ~153 lines. `Result._compute_delta_resonance()`
was not addressed.

## Motivation

Two methods and one class exceed reasonable complexity, violating the Single
Responsibility Principle:

| Method | Lines | Cyclomatic Complexity | Responsibilities |
|--------|-------|-----------------------|------------------|
| `Result._compute_delta_resonance()` | ~190 | ~15 | Shape normalization, spatial dim factorization (duplicated), physics calculation, error handling |
| `Measurement.fit_odmr()` | ~153 | ~12 | Model detection, validation, fitting, parameter extraction, metrics, result construction |
| `FitManager` (class) | ~462 | N/A | Constraint management, parameter guessing, fitting orchestration, result extraction |

These methods are difficult to test in isolation, hard to understand, and resist
modification. Adding a new model or output format requires understanding the entire
method.

## Specification

### 4A: Decompose `_compute_delta_resonance()` (~190 lines -> 4 methods)

Current method handles 6 shape branches with duplicated factorization logic.

```python
# New decomposition:

def _get_resonance_data(self) -> np.ndarray:
    """Extract and validate resonance frequency data from fit results."""

def _resolve_spatial_dims(self, n_pixels: int) -> tuple[int, int]:
    """Factorize pixel count into (height, width) spatial dimensions.

    Deduplicates the two copies of factorization logic currently in
    _compute_delta_resonance().
    """

def _calc_field_difference(
    self,
    resonance: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Compute magnetic field difference from resonance frequencies."""

def _compute_delta_resonance(self) -> np.ndarray:
    """Orchestrator: 10-line method calling the above."""
    resonance = self._get_resonance_data()
    h, w = self._resolve_spatial_dims(resonance.shape[-1])
    return self._calc_field_difference(resonance, h, w)
```

### 4B: Decompose `Measurement.fit_odmr()` (~150 lines -> 5 methods)

```python
def _detect_model(self, data: np.ndarray) -> str:
    """Determine appropriate ESR model from data characteristics."""

def _validate_fit_prerequisites(self) -> None:
    """Check that measurement data is loaded and valid for fitting."""

def _extract_fit_parameters(self, fit_manager: FitManager) -> dict:
    """Extract fitted parameters from FitManager into structured dict."""

def _compute_quality_metrics(self, parameters: dict) -> dict:
    """Compute fit quality metrics (chi-squared, R-squared, etc.)."""

def fit_odmr(self, ...) -> FitResult:
    """Orchestrator calling the above methods."""
```

### 4C: Split FitManager (~462 lines -> focused classes)

`ConstraintManager` is already partially extracted. Complete the separation:

1. **`ConstraintManager`** (already exists) - tighten its interface, ensure it
   owns all constraint logic currently duplicated in `FitManager`
2. **`ParameterGuesser`** - new class wrapping initial parameter estimation logic
   currently in `FitManager._guess_initial_params()` and related methods
3. **`FitManager`** - orchestrator that delegates to `ConstraintManager` and
   `ParameterGuesser`, handles pygpufit interaction

Target: No class exceeds 250 lines. No method exceeds 50 lines (excluding
docstrings).

## Files Affected

- `src/QDMpy/result.py` (decompose `_compute_delta_resonance`)
- `src/QDMpy/measurement.py` (decompose `fit_odmr`)
- `src/QDMpy/fit.py` (extract `ParameterGuesser`, clean `FitManager`)
- Tests for all three modules (add unit tests for new helper methods)

## Backwards Compatibility

No public API changes. `Measurement.fit_odmr()` and `Result._compute_delta_resonance()`
retain their signatures. Internal helper methods are private (prefixed with `_`).

## Verification

```bash
uv run pytest                              # All tests pass
uv run ruff check .                        # No lint errors
# Verify complexity targets:
uv run radon cc src/QDMpy/result.py -s     # No function > 10
uv run radon cc src/QDMpy/measurement.py -s
uv run radon cc src/QDMpy/fit.py -s
```

## Rejection Alternatives

**Alternative: Leave methods as-is with better comments.** Rejected because
comments don't solve testability. The 190-line method cannot be unit-tested for
its factorization logic without running the entire physics pipeline.

**Alternative: Decompose into free functions instead of methods.** Rejected
because the helper methods share state with their parent class (`self.fit_results`,
`self.odmr_data`, etc.). Methods are the natural decomposition.
