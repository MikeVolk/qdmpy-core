# QEP-016: Documentation Overhaul

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Priority** | P1 |
| **Complexity** | L |
| **Depends on** | QEP-007, QEP-009, QEP-011 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-17 |

## Motivation

The documentation is significantly out of date and incomplete. The ongoing
architectural overhaul (xarray adoption via QEP-011, Pydantic validation via
QEP-007, domain exceptions via QEP-009) has invalidated much of the existing
content. Specific problems:

1. **Quick start references dead APIs** — `QDMpy.ODMR.from_files()`,
   `odmr.process_data()`, `fit.get_param('center')` no longer match the current
   implementation.
2. **Tutorials are thin skeletons** — bullet-point outlines rather than
   worked examples with real output.
3. **No tutorial coverage for key subsystems** — data loading/validation,
   processors, fitting, models, results/export, and the CLI each deserve a
   standalone walkthrough.
4. **API reference is auto-generated stubs** — mkdocstrings pages exist but
   source docstrings are sparse, so the rendered output is incomplete.
5. **No architecture / design docs** — users and contributors have no high-level
   overview of how the pieces fit together.
6. **Installation docs still lead with `pip`** — should lead with `uv` per
   project conventions.
7. **Jupyter notebooks are orphaned** — `.ipynb` files in `docs/tutorials/` but
   not wired into the nav or kept current.

## Goals

- Every public module has a narrative tutorial **and** complete API reference.
- A new user can go from install to a fitted magnetic field map by following
  the quick start alone.
- A contributor can understand the architecture from the docs without reading
  source.
- All code examples are tested or at minimum runnable against the current API.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Specification

### 1. Restructure Navigation

```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Installation: installation.md
    - Quick Start: quickstart.md
  - Tutorials:
    - Overview: tutorials/index.md
    - Loading ODMR Data: tutorials/data.md
    - Processors: tutorials/processors.md
    - Fitting ODMR Spectra: tutorials/fitting.md
    - Spectral Models: tutorials/models.md
    - Working with Results: tutorials/results.md
    - Command Line Interface: tutorials/cli.md
  - Architecture:
    - Overview: architecture/overview.md
    - Data Flow: architecture/dataflow.md
  - API Reference:
    - Overview: api/index.md
    - Core:
      - FitManager: api/core/fit.md
      - Models: api/core/models.md
      - Measurement: api/core/measurement.md
      - FitResult: api/core/result.md
    - ODMR:
      - ODMR: api/odmr/odmr.md
      - ODMRData: api/odmr/data.md
      - Processors: api/odmr/processors.md
      - I/O: api/odmr/io.md
    - Utilities: api/utils.md
    - Constants: api/constants.md
    - Exceptions: api/exceptions.md
    - Settings: api/settings.md
  - Changelog: changelog.md
```

### 2. Tutorial: Loading ODMR Data (`tutorials/data.md`)

Covers the full data loading pipeline:

- Supported file formats (MATLAB `.mat`)
- Loading data with `ODMRData` / ODMR I/O
- Data shape conventions `(polarity, freq_range, y, x, freq_idx)` and what each
  dimension means physically
- Pydantic validation (QEP-007) — what gets validated on construction
- Inspecting data: frequencies, pixel dimensions, metadata
- Working with xarray DataArrays (QEP-011) — indexing by label, selecting
  polarities, slicing frequency ranges
- Common errors and how to fix them (`DataLoadError`, `DataValidationError`,
  `DataShapeError`)

### 3. Tutorial: Processors (`tutorials/processors.md`)

Rewrite from the current skeleton:

- What processors do and why (binning, normalization, global subtraction)
- Built-in processor catalogue with parameters
- Applying a processing pipeline to ODMR data
- Writing a custom processor
- Inspecting data before/after processing
- When to use which processor (decision guide)

### 4. Tutorial: Fitting ODMR Spectra (`tutorials/fitting.md`)

Rewrite from the current skeleton:

- Creating a `FitManager`
- Choosing a model (or using auto-guess)
- Running a fit
- Understanding convergence and `FitConvergenceError`
- Constraint system (`ConstraintManager`) — fixing, bounding, linking parameters
- Pixel-level vs global fits
- GPU acceleration with pyGpufit (when available)
- Interpreting fit quality metrics

### 5. Tutorial: Spectral Models (`tutorials/models.md`)

Update and expand:

- ESR14N (14-nitrogen, 3 dips per polarity)
- ESR15N (15-nitrogen, 2 dips per polarity)
- ESRSINGLE (single Lorentzian)
- Model parameter names and physical meaning
- Creating custom models (if supported)
- Relationship between model parameters and NV physics

### 6. Tutorial: Working with Results (`tutorials/results.md`)

New tutorial:

- `FitResult` Pydantic model — fields, validation
- Extracting parameter maps (center frequency, width, contrast)
- Computing derived quantities (B-field from splitting)
- Exporting results (formats, file naming)
- Plotting result maps

### 7. Tutorial: CLI (`tutorials/cli.md`)

Move and expand from `docs/cli.md`:

- Available commands and subcommands
- Processing a dataset end-to-end from the command line
- Common flags and options
- Scripting / batch processing

### 8. Architecture Documentation

New section under `docs/architecture/`:

- **`overview.md`** — high-level component diagram showing ODMR → Processors →
  FitManager → FitResult pipeline. Explain the role of each module.
- **`dataflow.md`** — how data flows from raw `.mat` files through processing
  to fitted parameter maps. Include the xarray dimension conventions and how
  polarities/frequencies are represented.

Leverage the existing Mermaid diagrams in `docs/diagrams/` (architecture,
data flow, class diagram, workflow sequence) — embed or link them.

### 9. Quick Start Rewrite (`quickstart.md`)

Replace the current broken example with a working end-to-end snippet that:

1. Installs with `uv`
2. Loads example data
3. Applies default processing
4. Fits with default model
5. Plots a center-frequency map

Every line must be runnable against the current API.

### 10. Installation Rewrite (`installation.md`)

- Lead with `uv` (primary)
- `pip` as alternative
- GPU support (pyGpufit) as opt-in
- Python version requirement (>=3.12)
- Verify installation command

### 11. API Reference Completeness

For every public module, ensure:

- mkdocstrings directive points to the correct import path
- Source code has Google-style docstrings with:
  - One-line summary
  - Args/Returns/Raises sections
  - At least one usage example for key classes/functions
- Add hand-written "Overview" paragraphs at the top of each API page explaining
  when and why to use the module

### 12. Clean Up Stale Content

- Remove or update `docs/api_docs.md` (duplicate of per-module API pages)
- Remove orphaned `.ipynb_checkpoints/` directory
- Remove `docs/tutorials/tutorial_old.ipynb` if no longer relevant
- Audit all existing `.ipynb` notebooks — either update or remove

## Files Affected

### New Files
- `docs/tutorials/data.md`
- `docs/tutorials/results.md`
- `docs/architecture/overview.md`
- `docs/architecture/dataflow.md`

### Modified Files
- `mkdocs.yml` (nav restructure)
- `docs/index.md` (update landing page)
- `docs/quickstart.md` (full rewrite)
- `docs/installation.md` (rewrite to lead with uv)
- `docs/tutorials/index.md` (update listing)
- `docs/tutorials/processors.md` (rewrite)
- `docs/tutorials/fitting.md` (rewrite)
- `docs/tutorials/models.md` (update for current API)
- `docs/cli.md` → `docs/tutorials/cli.md` (move + expand)
- `docs/api/index.md` (add overview text)
- `docs/api/core/*.md` (add overview paragraphs)
- `docs/api/odmr/*.md` (add overview paragraphs)
- All source files under `src/QDMpy/` (docstring completeness pass)

### Deleted Files
- `docs/tutorials/.ipynb_checkpoints/` (stale)
- `docs/tutorials/tutorial_old.ipynb` (stale)
- `docs/api_docs.md` (superseded by per-module pages)

## Implementation Order

1. **Phase 1 — Structure** : Restructure nav in `mkdocs.yml`, create stub files,
   clean up stale content.
2. **Phase 2 — Quick Start & Installation** : Rewrite these two pages first so
   new users have a working entry point.
3. **Phase 3 — Tutorials** : Write each tutorial in dependency order: data →
   processors → fitting → models → results → CLI.
4. **Phase 4 — Architecture Docs** : Write overview and dataflow pages, embed
   Mermaid diagrams.
5. **Phase 5 — API Reference** : Docstring completeness pass across all source
   modules, add overview paragraphs to API pages.
6. **Phase 6 — Verification** : Build docs locally (`mkdocs serve`), verify all
   links, test code examples.

## Migration Notes

- Existing bookmarks to `docs/cli.md` will break when moved to
  `docs/tutorials/cli.md` — this is acceptable since the docs are not yet
  widely deployed.
- No Python API changes are required by this QEP.
- Source docstring additions are additive and backwards-compatible.

## Backwards Compatibility

No impact. This QEP only modifies documentation and docstrings.

## Verification

```bash
# Build docs and check for warnings
uv run mkdocs build --strict

# Serve locally and spot-check all pages
uv run mkdocs serve

# Verify no broken internal links
uv run mkdocs build --strict 2>&1 | grep -i "warning"

# Verify docstring coverage (optional, if tool available)
uv run interrogate src/QDMpy/ -v
```

## Rejection Alternatives

**Alternative: Use Sphinx instead of MkDocs.** Rejected. MkDocs with Material
is already configured and working. Switching documentation frameworks is
unnecessary churn that doesn't improve content quality.

**Alternative: Write tutorials as Jupyter notebooks only.** Rejected. Markdown
tutorials are easier to maintain, review in PRs, and render consistently.
Notebooks can supplement tutorials but should not replace them — they rot
faster and create merge conflicts.

**Alternative: Defer until all architectural QEPs are complete.** Rejected.
The documentation is already significantly outdated and QEP-007, QEP-009, and
QEP-011 (the major API-changing proposals) are implemented. Waiting longer
only widens the gap.
