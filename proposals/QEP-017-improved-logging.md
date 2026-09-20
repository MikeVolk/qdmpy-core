# QEP-017: Improved Logging

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | S |
| **Depends on** | QEP-009 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-17 |

## Motivation

CLAUDE.md mandates "use logging extensively (loguru not stdlib logging)", but
several modules have zero logging and the configuration is minimal:

1. **Silent I/O modules** — `io.py` and `odmr/io.py` perform file loading
   (the most common failure point) with no log output.  Users get exceptions
   but no trace of what was attempted.
2. **No file sink** — all output goes to stderr only.  Long-running batch
   jobs have no persistent log.
3. **No performance timing** — expensive operations (MATLAB loading, fitting)
   log start/finish but not duration consistently.
4. **Inconsistent format strings** — mix of f-strings and %-formatting in
   logger calls.


## Specification

### 1. Add logging to silent modules

Add `from loguru import logger` and appropriate log calls to:

- **`io.py`** — log image file selection and load results
- **`odmr/io.py`** — log each MATLAB file load, data shapes, frequency
  extraction
- **`models.py`** — log model registration

### 2. Add optional file sink

Extend `LoggingSettings` with an optional `log_file` path.  When set,
`_configure_logging` adds a second sink writing to that file with rotation.

```python
class LoggingSettings(BaseModel):
    log_level: Literal[...] = "WARNING"
    log_file: str | None = None
```

### 3. Standardize on f-strings

Replace any `logger.debug("msg %s", val)` calls with `logger.debug(f"msg {val}")`
for consistency (loguru optimizes lazy f-string evaluation).

### 4. Add timing to data loading

Add `logger.info` calls with elapsed time for MATLAB file loading in
`odmr/io.py`.

## Files Affected

- `src/QDMpy/io.py` (add logging)
- `src/QDMpy/odmr/io.py` (add logging + timing)
- `src/QDMpy/models.py` (expand logging)
- `src/QDMpy/settings.py` (add `log_file` field)
- `src/QDMpy/__init__.py` (file sink in `_configure_logging`)
- `src/QDMpy/fit.py` (standardize format strings)

## Backwards Compatibility

Fully backwards compatible. The `log_file` field defaults to `None` (no file
sink), preserving current behavior.  All changes are additive log statements.

## Verification

```bash
# Run tests to ensure nothing breaks
uv run pytest

# Verify logging output at DEBUG level
QDMPY_LOG_LEVEL=DEBUG uv run python -c "from QDMpy.odmr.io import MatlabLoader; print('ok')"

# Lint
uv run ruff check src/QDMpy/io.py src/QDMpy/odmr/io.py src/QDMpy/settings.py
```

## Rejection Alternatives

**Alternative: Use structlog instead of loguru.** Rejected. Loguru is already
the project standard and provides structured binding via `logger.bind()`.
Switching libraries is unnecessary churn.

**Alternative: Add logging to every function.** Rejected. Only add logging
where it provides diagnostic value — I/O boundaries, expensive operations,
and decision points. Utility math functions do not need logging.
