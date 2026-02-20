---
paths:
  - "**/*.py"
  - "**/*.pyi"
---
# Python Coding Style

> This file extends [common/coding-style.md](../common/coding-style.md) with Python specific content.

## Standards

- Follow **PEP 8** conventions
- Use **type annotations** on all function signatures

## Immutability

Prefer immutable data structures:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class User:
    name: str
    email: str

from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
```

## Formatting

- **ruff format** for code formatting (replaces black)
- **ruff check --select I** for import sorting (replaces isort)
- **ruff check** for linting

## Type Checking

- **ty** for static type checking (replaces mypy)
- Run as: `uv run ty src/QDMpy`

## Reference

See skill: `python-patterns` for comprehensive Python idioms and patterns.
