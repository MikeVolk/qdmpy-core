# QEP-002: Eliminate Global State and sys.path Hacks

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P1 |
| **Complexity** | M |
| **Depends on** | Nothing (QEP-001 superseded by QEP-011) |
| **Blocks** | QEP-006 |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |

## Motivation

The current codebase has several global state issues that cause side effects at
import time and make testing fragile:

1. **5 `sys.path` hacks** across `guess.py`, `measurement.py`, `odmr/data.py`,
   `odmr/odmr.py`, and `__init__.py`. These modify the global interpreter state
   and are unnecessary with proper package installation.

2. **`SETTINGS` singleton** created at import time in `__init__.py`. This triggers
   config file creation and Pydantic validation on every import, even in tests that
   don't need settings.

3. **`PYGPUFIT_PRESENT` global flag** that forces a CUDA availability check on
   every import. Slows imports and fails noisily in environments without GPU.

4. **`logger.info("WELCOME TO QDMpy")`** fires on every import, polluting test
   output and logs.

5. **`ModelRegistry._registry`** is mutable class-level state populated via side
   effects during module import.

These violate the Dependency Inversion Principle and make unit testing require
heavy mocking of module-level state.


## Specification

### 1. Delete All sys.path Hacks

Remove the following pattern from all files:

```python
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Files: `guess.py`, `measurement.py`, `odmr/data.py`, `odmr/odmr.py`

Remove `sys.path.append(str(SRC_PATH))` from `__init__.py`.

Delete `setup_package_paths()` from `utils.py` and its call in `models.py`.

### 2. Lazy Settings Accessor

Replace the eager singleton with a lazy accessor:

```python
# src/QDMpy/__init__.py
_settings: QDMpySettings | None = None

def get_settings() -> QDMpySettings:
    """Get application settings, creating on first access."""
    global _settings
    if _settings is None:
        _settings = QDMpySettings()
    return _settings

def reset_settings() -> None:
    """Reset settings to default. Useful for testing."""
    global _settings
    _settings = None
```

Update all consumers to use `get_settings()` instead of importing `SETTINGS`:
- `src/QDMpy/fit.py`
- `src/QDMpy/models.py`
- Any other module importing `SETTINGS`

### 3. Lazy PYGPUFIT Check

Replace the eager global with a cached function:

```python
# src/QDMpy/__init__.py
from functools import cache

@cache
def is_pygpufit_available() -> bool:
    """Check if pygpufit is available. Result is cached after first call."""
    try:
        import pygpufit.gpufit as gf
        return True
    except ImportError:
        return False
```

Update `fit.py` to use `is_pygpufit_available()` instead of `PYGPUFIT_PRESENT`.

### 4. Remove Welcome Message

Delete `logger.info("WELCOME TO QDMpy")` from `__init__.py`. If a welcome message
is desired, it should be in the CLI entry point only.

### 5. Clean __init__.py

The cleaned `__init__.py` should contain only:
- `__version__` definition
- Path constants (`SRC_PATH`, `PACKAGE_PATH`, etc.)
- Lazy accessor functions (`get_settings`, `is_pygpufit_available`)
- Public API exports via `__all__`
- No side effects at import time

## Files Affected

- `src/QDMpy/__init__.py` (major refactor)
- `src/QDMpy/utils.py` (remove `setup_package_paths`)
- `src/QDMpy/models.py` (remove `setup_package_paths` call, use `get_settings`)
- `src/QDMpy/guess.py` (remove sys.path hack)
- `src/QDMpy/measurement.py` (remove sys.path hack)
- `src/QDMpy/odmr/data.py` (remove sys.path hack)
- `src/QDMpy/odmr/odmr.py` (remove sys.path hack)
- `src/QDMpy/fit.py` (use `get_settings()`, `is_pygpufit_available()`)
- Tests that import `SETTINGS` or `PYGPUFIT_PRESENT` directly

## Backwards Compatibility

- `SETTINGS` and `PYGPUFIT_PRESENT` will be removed from public API
- `get_settings()` and `is_pygpufit_available()` are the replacements
- A deprecation period is not necessary since this is an internal API used only
  within the package itself

## Verification

```bash
uv run pytest                    # All tests still pass
uv run python -c "import QDMpy"  # No output, no side effects
uv run ruff check .              # No lint errors
uv run mypy src/QDMpy            # No type errors
```

## Rejection Alternatives

**Alternative: Keep SETTINGS but make it lazy via `__getattr__`.** Rejected because
module-level `__getattr__` is harder to type-check and understand. An explicit
function is clearer.
