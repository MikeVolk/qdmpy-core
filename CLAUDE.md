# QDMpy Development Guide

## Build & Development
- Install: `uv pip install -e .`
- Create venv: `uv venv`
- Activate venv: `source .venv/bin/activate`

## Testing
- Run all tests: `pytest`
- Run single test: `pytest tests/test_file.py::test_function -v`
- Run with coverage: `pytest --cov=QDMpy --cov-report=term-missing`

## Linting
- Run all checks: `pre-commit run --all-files`
- Run ruff: `ruff check .`
- Run mypy: `mypy src/QDMpy`

## Code Style
- Python >=3.12
- 80 char line length (strict PEP8)
- Google style docstrings
- Type annotations required for all functions
- Snake_case for variables/functions, PascalCase for classes
- Import order: FUTURE → STDLIB → THIRDPARTY → FIRSTPARTY → LOCALFOLDER
- Use single quotes for strings
- Raise specific exceptions (see exceptions.py)
- Max complexity: 10 (cyclomatic), 8 (cognitive)