# QDMpy Development Guide

The `claude` branch acts as main/master. NEVER commit directly to `claude` or `main`/`master`. Always create a feature branch from `claude`, do all work there, then merge back into `claude` when finished.

## Principles
- Follow **clean code** principles: meaningful names, small focused functions, single responsibility, DRY, no dead code, minimal comments (code should be self-documenting)
- Always use **uv** as the package manager (never pip, poetry, or conda directly)

## Reference Codebase
- The pre-overhaul working repo lives at `~/git/QDMpy_old` — use it as a reference for existing behavior and logic during the architectural overhaul
- The MATLAB reference toolbox lives at `~/git/QDMlab` — use it to verify physics conventions and expected output

## Typical Data
- Image resolution: ~2k x 2k pixels (e.g. 1200×1920)
- Frequency sweeps: ~50 frequencies per range
- Polarities: 2 (negative applied field, positive applied field)
- Frequency ranges: 2 (low below ZFS ~2.87 GHz, high above ZFS)
- Internal 5D array shape: `(n_pol, n_frange, y, x, freq_idx)` — always this dim order
- Flat (for fitting): `(n_pol, n_frange, n_pixel, freq_idx)`

## Data & Polarity Conventions
- **`pol_0`** = negative applied bias field = `run_00000.mat` (first file alphabetically)
- **`pol_1`** = positive applied bias field = `run_00001.mat`
- **`frange_0`** = low-frequency branch (below ZFS, ~2.72–2.87 GHz)
- **`frange_1`** = high-frequency branch (above ZFS, ~2.87–3.02 GHz)
- **All internal frequencies are in GHz.** Hz↔GHz conversion only at pygpufit boundary in `fit.py`
- Diamond type: 14N is standard (3 hyperfine dips, model `ESR14N`, gpufit model_id=13)

## B111 Physics
The field along the NV [111] axis is extracted from the splitting between the two frequency branches:

```
dB[pol] = (R_high[pol] - R_low[pol]) / 2 / γ    # always positive, in µT
```

where `γ = GAMMA_NV = 28.024 GHz/T`. Flipping the bias sign separates remanent from induced:

```
negDiff = -dB[pol_0]    # negative value (pol_0 = neg field, sign convention)
posDiff = +dB[pol_1]    # positive value

b111_remanent = (negDiff + posDiff) / 2    # ferro / permanent magnetization
b111_induced  = (negDiff - posDiff) / 2    # para / bias-tracking (sign-flipped vs MATLAB B111para)
```

In `result.py`, `delta_resonance` has shape `(n_pol, 2, H, W)` where axis-1 is an artificial ±sign
dimension: `[:, 0]` = negatively-signed dB, `[:, 1]` = positively-signed dB. Extract as:
- `neg_diff = delta_res[0, 0]`  — pol_0 with d=−1
- `pos_diff = delta_res[-1, 1]` — pol_1 with d=+1

## Build & Development
- Install: `uv pip install -e .`
- Create venv: `uv venv`
- Activate venv: `source .venv/bin/activate`
- Run python commands: `uv run python`
- Add dependencies: `uv add <package>`

## Testing
- Run all tests: `uv run pytest`
- Run single test: `uv run pytest tests/test_file.py::test_function -v`
- Run with coverage: `uv run pytest --cov=QDMpy --cov-report=term-missing`

## Linting
- Run all checks: `pre-commit run --all-files`
- Run ruff: `uv run ruff check .`
- Run ty: `uv run ty src/QDMpy`

## Code Style
- Python >=3.12
- 100 char line length (strict PEP8)
- Google style docstrings
- Type annotations required for all functions
- Snake_case for variables/functions, PascalCase for classes
- Import order: FUTURE -> STDLIB -> THIRDPARTY -> FIRSTPARTY -> LOCALFOLDER
- Use single quotes for strings
- Raise specific exceptions (see exceptions.py)
- Max complexity: 10 (cyclomatic), 8 (cognitive)
- use logging extensively (logruru not stdlib logging)
- do not include a "authored by claude *" in commits

## Session Workflow
- **No summary files** — do not create session summaries or recap documents after work
- **Proposals required** — before any significant implementation, write a proposal document (PEP-style) under `/proposals/` with rationale, design, alternatives considered, and migration plan. Get approval before coding.
- **Changelog** — keep `CHANGELOG.md` up to date; add entries under `## [Unreleased]` as work is done (Added / Changed / Fixed / Performance / Removed); use the QEP tag when applicable

## Memory Files (`/memory/`)
- **Session start & post-compaction**: read `memory/index.md` first, then any topic files relevant to the current task
- **Session end**: update memory files to reflect any architectural changes, new patterns, renamed/moved modules, or discovered invariants made during the session
- Files: `index.md` (entry point), `architecture.md` (module graph + data flow), `odmr_module.md` (odmr/ subpackage), `fitting.md` (models/guess/fit/result), `core.md` (measurement/io/settings/exceptions/constants)
- Keep entries concise and LLM-readable; prefer mermaid diagrams over prose for flows
- Do **not** record session-specific state or in-progress work — only stable, confirmed facts
