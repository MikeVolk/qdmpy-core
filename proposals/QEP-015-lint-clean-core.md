# QEP-015: Achieve Clean Linting on Core Package

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | S |
| **Depends on** | QEP-014 (type safety fixes overlap) |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-16 |

## Motivation

`ruff check` reports **74 errors** in core source files (excluding notebooks
and tests). While 41 of these are `TRY003` (raise-vanilla-args, addressed by
QEP-009's exception hierarchy), the remaining **33 errors** include real code
quality issues:

| Rule | Count | Description |
|------|-------|-------------|
| TRY003 | 41 | Long messages in `raise` — needs domain exceptions (QEP-009) |
| PLR2004 | 13 | Magic values in comparisons |
| ANN102 | 4 | Missing `cls` type annotation |
| ANN401 | 3 | `Any` type annotations |
| ARG003 | 3 | Unused class method arguments |
| F821 | 2 | Undefined names (plotting — QEP-012) |
| PLW0603 | 2 | Global statement usage |
| ANN003 | 1 | Missing `**kwargs` type annotation |
| D105 | 1 | Undocumented magic method |
| PLR0913 | 1 | Too many arguments |
| RUF003 | 1 | Ambiguous unicode character |
| TRY300 | 1 | Consider `else` block in `try` |
| TRY301 | 1 | `raise` within `try` |

**The goal**: zero ruff errors on `src/QDMpy/` (excluding `plotting.py`).

The `TRY003` errors are the most numerous (41) and are directly tied to QEP-009.
This QEP focuses on the non-TRY003 issues. Once QEP-009 is implemented, TRY003
errors resolve naturally.


## Specification

### 1. Eliminate magic values (PLR2004 — 13 occurrences)

These are dimension-size comparisons scattered through `result.py`, `odmr/io.py`,
and `odmr/processors.py`:

```python
# result.py — dimension branching
if resonance.ndim == 4:  # PLR2004
if resonance.ndim == 3:  # PLR2004
if n_frange >= 2:        # PLR2004
```

**Fix:** These are inherent to array dimension checking in scientific code.
The appropriate fix depends on context:

For dimension checks, suppress with `noqa` — these are fundamental to array
programming and named constants would reduce readability:

```python
if resonance.ndim == 4:  # noqa: PLR2004 — standard ndarray dimension check
```

For `odmr/io.py` imgStack counts, extract a constant:

```python
DUAL_POLARITY_STACKS = 2
QUAD_POLARITY_STACKS = 4

if n_img_stacks == DUAL_POLARITY_STACKS:
    ...
elif n_img_stacks == QUAD_POLARITY_STACKS:
    ...
```

For `processors.py` threshold (`0.001`), move to settings or a module constant:

```python
FLUORESCENCE_DELTA_THRESHOLD = 0.001
```

### 2. Fix annotation gaps (ANN102, ANN401, ANN003 — 8 occurrences)

```python
# ANN102: Missing cls type annotation (4 occurrences in ModelRegistry)
@classmethod
def register(cls, model_cls):  # missing: cls: type[ModelRegistry]

# ANN401: Any type (3 occurrences)
# Review each — replace with specific type or add justification comment

# ANN003: Missing **kwargs annotation
def func(**kwargs):  # missing: **kwargs: Any
```

### 3. Remove unused arguments (ARG003 — 3 occurrences)

Three class methods accept arguments they don't use. Either:
- Remove the argument if callers don't pass it
- Add `_` prefix if required by an interface
- Actually use the argument if it was intended

### 4. Replace global statements (PLW0603 — 2 occurrences)

Both are in `__init__.py` for the `_settings` singleton:

```python
# Before
def get_settings() -> QDMpySettings:
    global _settings
    if _settings is None:
        ...

# After — use a module-level mutable container
_state: dict[str, QDMpySettings | None] = {'settings': None}

def get_settings() -> QDMpySettings:
    if _state['settings'] is None:
        ...
```

Or simply suppress — `global` for a module-level singleton is a well-understood
pattern and the ruff rule is overly strict here.

### 5. Fix remaining minor issues

- **D105** (1): Add docstring to `__repr__` or `__str__` magic method
- **RUF003** (1): Replace ambiguous unicode character (likely a mu or degree symbol
  in a comment — use ASCII equivalent or add `noqa`)
- **TRY300** (1): Move return from `try` to `else` block
- **TRY301** (1): Move `raise` from inside `try` to after it
- **PLR0913** (1): Too many function arguments — consider a config dataclass or
  suppress if the arguments are all necessary

## Approach to Suppression

Some rules are counterproductive for scientific code:
- **PLR2004 for ndim checks**: `if arr.ndim == 3` is universally understood in
  numpy code. A named constant would be absurd.
- **PLW0603 for singleton**: The `global` keyword for a module singleton is
  standard Python.

For these, add targeted `noqa` comments with explanation. Do **not** disable
the rules project-wide — they catch real issues elsewhere.

## Files Affected

- `src/QDMpy/__init__.py` — global statement
- `src/QDMpy/result.py` — magic values, annotations
- `src/QDMpy/odmr/io.py` — magic values
- `src/QDMpy/odmr/processors.py` — magic values, threshold constant
- `src/QDMpy/models.py` — annotations, unused args
- `src/QDMpy/fit.py` — annotations
- `src/QDMpy/settings.py` — annotations

## Verification

```bash
# Zero non-TRY003 ruff errors in core
uv run ruff check src/QDMpy/ --ignore TRY003 --exclude src/QDMpy/plotting.py
# Expected: All checks passed!

# Full ruff (will still show TRY003 until QEP-009)
uv run ruff check src/QDMpy/ --exclude src/QDMpy/plotting.py
# Expected: only TRY003 errors remain
```

## Rejection Alternatives

**Alternative: Disable problematic rules in pyproject.toml.** Rejected for most
rules — they catch real issues. Better to fix or `noqa` with explanation on a
case-by-case basis.

**Alternative: Fix TRY003 here too.** Rejected — TRY003 requires the full
exception hierarchy from QEP-009. Fixing TRY003 without domain exceptions would
mean adding empty exception subclasses just to satisfy the linter.

**Alternative: Do this as part of QEP-014.** The type safety (mypy) and lint
(ruff) fixes overlap slightly but address different tools and concerns. Keeping
them separate allows independent review and implementation.
