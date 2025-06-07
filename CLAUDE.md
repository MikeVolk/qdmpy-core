# QDMpy Development Guide

## Build & Development
- Install: `uv pip install -e .`
- Create venv: `uv venv`
- Activate venv: `source .venv/bin/activate`

## running
- Run python commands: `uv run python`
## Testing
- Run all tests: `uv run pytest`
- Run single test: `uv run pytest tests/test_file.py::test_function -v`
- Run with coverage: `uv run pytest --cov=QDMpy --cov-report=term-missing`

## Linting
- Run all checks: `pre-commit run --all-files`
- Run ruff: `uv run ruff check .`
- Run mypy: `uv run mypy src/QDMpy`

## Code Style
- Python >=3.12
- 100 char line length (strict PEP8)
- Google style docstrings
- Type annotations required for all functions
- Snake_case for variables/functions, PascalCase for classes
- Import order: FUTURE → STDLIB → THIRDPARTY → FIRSTPARTY → LOCALFOLDER
- Use single quotes for strings
- Raise specific exceptions (see exceptions.py)
- Max complexity: 10 (cyclomatic), 8 (cognitive)