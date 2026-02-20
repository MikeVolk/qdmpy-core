---
paths:
  - "**/*.py"
  - "**/*.pyi"
---
# QDMpy-Specific Overrides

> Project-specific rules that extend and override the generic Python rules.

## Tooling — always prefix with `uv run`

```bash
uv run pytest                          # run tests
uv run pytest --cov=QDMpy --cov-report=term-missing   # with coverage
uv run ruff check .                    # lint
uv run ruff format .                   # format
uv run ruff check --select I .         # import order
uv run ty src/QDMpy                    # type check
pre-commit run --all-files             # full pre-commit suite
```

Never invoke `pytest`, `ruff`, `ty`, or `mypy` without the `uv run` prefix in this project.

## Branch conventions

- **Working branch**: `claude` — treat it as `main`. Never merge into `main`/`master`.
- New work: branch from `claude`, merge back into `claude`.
- Never commit without being explicitly asked.

## Frequency units

- **All internal frequencies are in GHz.** No Hz conversions inside the library.
- Hz appears only at the `pygpufit` GPU kernel boundary (currently even that stays GHz).

## Coordinate labels

- Polarity: `neg` / `pos` (not `pol_0` / `pol_1`)
- Frequency range: `low` / `high` (not `frange_0` / `frange_1`)

## No summary files

Do not create session-summary or recap documents. Update `CHANGELOG.md` instead.

## Proposals required

Before any significant implementation, write a PEP-style proposal under `/proposals/`
and get approval before writing code.

## Logging

Use `loguru` (not stdlib `logging`) throughout the codebase.

## Package manager

Use `uv` exclusively. Never `pip`, `poetry`, or `conda`.
