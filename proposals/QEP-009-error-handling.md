# QEP-009: Improve Error Handling

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P3 |
| **Complexity** | S |
| **Depends on** | QEP-007 (QEP-001 superseded by QEP-011) |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |

## Motivation

The codebase has only 3 custom exceptions but uses generic `ValueError`,
`RuntimeError`, and `ImportError` for domain-specific errors. This makes it
difficult for users to:

1. Catch specific error types (e.g., distinguish a fitting failure from a data
   loading failure)
2. Understand what went wrong from the exception type alone
3. Build robust pipelines with targeted error handling

Current state of `exceptions.py`:

```python
class QDMpyError(Exception): ...
class CantImportError(QDMpyError): ...
class FitNotPerformedError(QDMpyError): ...
```

Examples of generic exceptions used for domain errors:

```python
# fit.py
raise ValueError("Model not found")  # Should be ModelNotFoundError
raise RuntimeError("Fit did not converge")  # Should be FitConvergenceError

# odmr/data.py
raise ValueError("Invalid data shape")  # Should be DataShapeError

# measurement.py
raise ValueError("No data loaded")  # Should be DataLoadError
```


## QEP-011 Impact

No overlap. QEP-011 did not introduce any domain-specific exceptions. Generic
`ValueError`/`RuntimeError` are still used throughout for domain errors.
The xarray adoption may surface new error scenarios (e.g., missing coordinates,
wrong dimension names) that would benefit from specific exception types.

## Specification

### 1. Expanded Exception Hierarchy

```python
# src/QDMpy/exceptions.py

class QDMpyError(Exception):
    """Base exception for all QDMpy errors."""


# --- Data Errors ---

class DataError(QDMpyError):
    """Base for data-related errors."""

class DataLoadError(DataError):
    """Failed to load data from file or source."""

class DataValidationError(DataError):
    """Data failed validation checks."""

class DataShapeError(DataError):
    """Data array has unexpected shape or dimensions."""


# --- Fitting Errors ---

class FittingError(QDMpyError):
    """Base for fitting-related errors."""

class FitNotPerformedError(FittingError):
    """Attempted to access fit results before fitting."""

class FitConvergenceError(FittingError):
    """Fit did not converge within allowed iterations."""

class ModelNotFoundError(FittingError):
    """Requested model is not registered."""

class ModelGuessNotPossible(FittingError):
    """Cannot determine appropriate model from data."""


# --- Configuration Errors ---

class ConfigurationError(QDMpyError):
    """Invalid or missing configuration."""


# --- Dependency Errors ---

class DependencyError(QDMpyError):
    """Required dependency is not available."""
```

### 2. Replace Generic Exceptions

Systematically replace generic exceptions with specific ones:

| File | Current | Replacement |
|------|---------|-------------|
| `fit.py` | `ValueError("Model not found")` | `ModelNotFoundError(name)` |
| `fit.py` | `RuntimeError("Fit did not converge")` | `FitConvergenceError(details)` |
| `odmr/data.py` | `ValueError("Invalid data shape")` | `DataShapeError(expected, got)` |
| `measurement.py` | `ValueError("No data loaded")` | `DataLoadError("No data loaded")` |
| `guess.py` | `ValueError("Cannot determine model")` | `ModelGuessNotPossible(reason)` |
| Various | `ImportError(...)` | `DependencyError(package, reason)` |

### 3. Deprecate CantImportError

`CantImportError` is replaced by `DependencyError`. Keep as alias for one release:

```python
# Deprecated alias
CantImportError = DependencyError
```

### 4. Error Message Quality

All custom exceptions should include actionable context:

```python
# Bad
raise DataShapeError("Wrong shape")

# Good
raise DataShapeError(
    f"Expected DataArray with dims (polarity, freq_range, y, x, freq_idx), "
    f"got dims {data.dims} with sizes {dict(data.sizes)}"
)
```

## Files Affected

- `src/QDMpy/exceptions.py` (expand hierarchy)
- `src/QDMpy/fit.py` (replace generic exceptions)
- `src/QDMpy/result.py` (replace generic exceptions)
- `src/QDMpy/odmr/data.py` (replace generic exceptions)
- `src/QDMpy/measurement.py` (replace generic exceptions)
- `src/QDMpy/guess.py` (replace generic exceptions)
- Tests that assert specific exception types

## Backwards Compatibility

- `QDMpyError` base class is preserved — existing `except QDMpyError` catches
  will continue to work
- `FitNotPerformedError` is preserved (now inherits from `FittingError` instead
  of `QDMpyError` directly, but `QDMpyError` catch still works)
- `CantImportError` kept as deprecated alias for `DependencyError`
- Code that catches generic `ValueError` or `RuntimeError` for QDMpy errors will
  need updating — this is intentional to improve error handling specificity

## Verification

```bash
uv run pytest
uv run ruff check .
# Verify no generic exceptions for domain errors:
grep -rn "raise ValueError\|raise RuntimeError" src/QDMpy/ | grep -v "__pycache__"
# Should show minimal results (only for truly generic value/runtime errors)
```

## Rejection Alternatives

**Alternative: Use error codes instead of exception hierarchy.** Rejected because
Python's exception hierarchy is the idiomatic approach. Error codes require
checking at every call site and lose the benefit of `try/except` specificity.

**Alternative: Deeper hierarchy (e.g., separate FileNotFoundError subclass).**
Rejected as over-engineering. The proposed hierarchy has 3 levels which is
sufficient for the domain complexity.
