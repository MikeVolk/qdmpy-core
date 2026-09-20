# QEP-012: Fix and Restructure the Plotting Module

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P4 |
| **Complexity** | L |
| **Depends on** | None |
| **Blocks** | QEP-010 (CLI/Plotting coverage) |
| **Author** | QDMpy Team |
| **Created** | 2026-02-16 |

## Motivation

The plotting layer still carries substantial technical debt after the module split
to `src/qdmpy/plotting/`. Runtime behavior is mostly functional, but layout and
visual consistency are still uneven across plot entry points.

Current issues:

**Layout and rendering quality issues:**
- Some map plots still show colorbar geometry that does not visually match the
  target image axes after layout adjustments.
- Mixed use of `plt.tight_layout()`/`fig.tight_layout()` with `suptitle` and
  appended colorbar axes causes occasional clipping, overlap, or uneven panel sizing.
- Multi-panel overview plots can produce inconsistent spacing between rows/columns,
  especially when a subset of panels is hidden.
- Spatial map styling is repeated in multiple modules (`fit.py`, `display.py`,
  `fields.py`, `odmr.py`) with slightly different defaults.

**Semantic consistency issues:**
- Unit labels are not uniformly accurate across plots (e.g. some fit parameter labels
  still use Hz text even though core frequencies are GHz).
- Title and axis label conventions are not consistently applied.

**Design problems:**
- Repeated map-rendering logic instead of one canonical helper API.
- Tight coupling between data selection, styling, and rendering in the same functions.
- Layout behavior is not tested directly (no assertions for colorbar/axes geometry).

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Specification

### Phase 1: Fix Runtime Errors (make the module importable and functional)

#### 1a. Fix `update_line()` and `update_marker()` signatures

```python
# Before (broken)
def update_line(ax, x, y=None, line=None) -> ...:
    (line,) = ax.plot(x, y, **plt_props)  # F821: plt_props undefined

# After (fixed)
def update_line(ax, x, y=None, line=None, **plt_props: Any) -> ...:
    (line,) = ax.plot(x, y, **plt_props)
```

#### 1b. Fix `update_img()` type annotation

```python
# Before
def update_img(ax, img, data, **plt_props):  # no-untyped-def

# After
def update_img(ax: plt.Axes, img: mpl.image.AxesImage | None,
               data: np.ndarray, **plt_props: Any) -> mpl.image.AxesImage:
```

#### 1c. Fix `savefig()` type

```python
# Before
if save:
    f.savefig(save)  # save is str | bool, but savefig needs str | Path

# After
if save and isinstance(save, str):
    f.savefig(save)
```

### Phase 2: Update to Current API

The functions `check_fit_pixel()`, `plot_fit_params()`, and related functions
reference an API that no longer exists. These must be rewritten to use the current
`Measurement`, `ODMR`, and `FitResult` interfaces.

#### 2a. Audit current API surface

Read the current public interfaces of:
- `Measurement` (measurement.py) — properties and methods available
- `ODMR` (odmr/odmr.py) — data access patterns
- `FitResult` (result.py) — parameter access, field calculations

#### 2b. Rewrite `check_fit_pixel()` against current API

The function should use:
- `FitResult.parameters` dict for fitted params
- `FitResult.model_name` for model identification
- `ODMR.processed_data` (xarray DataArray) for raw spectra
- Model functions from `models.py` for generating fit curves

#### 2c. Rewrite `plot_fit_params()` against current API

Replace `qdm_obj.get_param(param)` with `FitResult.parameters[param]` or a
method on `FitResult` that returns reshaped parameter maps.

#### 2d. Fix remaining functions that reference old API

Audit every function that takes `Measurement` or uses `qdm_obj.odmr.*` and update
to the current interface.

### Phase 3: Structural Cleanup

#### 3a. Extract a `PlotConfig` dataclass

Consolidate repeated styling parameters into a configuration object:

```python
@dataclass
class PlotConfig:
    figsize: tuple[float, float] = (10, 8)
    cmap: str = 'RdBu_r'
    origin: str = 'lower'
    dpi: int = 150
    save: bool = False
    filename: str | None = None
```

#### 3b. Reduce function count through parameterization

Many functions differ only in which data they plot. Consolidate:

```python
# Before: 4 near-identical functions
def plot_fit_result_field_map(result, ...)
def plot_fit_result_parameter_map(result, param, ...)
def plot_field_map(data, ...)
def plot_fit_params(qdm_obj, param, ...)

# After: 1 function + enum
def plot_spatial_map(data: NDArray, config: PlotConfig, ...) -> Figure:
    """Plot any 2D spatial map with consistent styling."""
```

#### 3c. Separate data preparation from rendering

Each plot function currently does data reshaping, normalization, and rendering
in a single function. Split into:
- Data preparation (pure functions returning arrays)
- Rendering (thin wrappers around matplotlib)

This enables unit testing the data logic without matplotlib.

### Phase 4: Layout and Visual Consistency Hardening

#### 4a. Standardize colorbar geometry behavior

- Keep `_add_colorbar()` as the single entry point for image colorbars.
- Ensure all image-map functions use it (no direct `fig.colorbar(..., ax=...)`
  variants that reintroduce inconsistent sizing).
- Add optional helper parameters (`size`, `pad`) only if needed globally; avoid
  one-off per-function overrides.

#### 4b. Normalize figure layout strategy

- Adopt one layout strategy for each figure type (simple single-panel map,
  multi-panel overview, diagnostics).
- Ensure suptitle and colorbar axes do not overlap after final layout pass.
- Remove ad-hoc spacing behavior when hiding unused subplots.

#### 4c. Enforce plot metadata consistency

- Align frequency-related labels to GHz convention where applicable.
- Standardize axis label text (`x [µm]`, `y [µm]`) and title style.

#### 4d. Add layout regression tests

- Add tests that render representative map figures and assert colorbar-axes height
  matches the corresponding image axes to within a small tolerance after draw.
- Add tests that verify no overlap/clipping regressions for overview plots with
  suptitle + colorbars.

## Files Affected

- `src/qdmpy/plotting/_common.py` — shared colorbar/layout helpers
- `src/qdmpy/plotting/fit.py` — map and overview plots
- `src/qdmpy/plotting/display.py` — dashboard-style overview layout
- `src/qdmpy/plotting/fields.py` — magnetic component map layout
- `src/qdmpy/plotting/odmr.py` — folding/diagnostic panel consistency
- `tests/test_plotting.py` — smoke + layout regression checks

## Verification

```bash
# Lint + type checks on plotting modules
uv run ruff check src/qdmpy/plotting/ tests/test_plotting.py
uv run ty check src/qdmpy/plotting/

# Plot behavior and layout regressions
uv run pytest tests/test_plotting.py -v

# Optional targeted coverage check
uv run pytest tests/test_plotting.py --cov=qdmpy.plotting --cov-report=term-missing
```

## Rejection Alternatives

**Alternative: Delete plotting.py entirely and start fresh.** Rejected because
the low-level helper functions (`plot_field_map`, `update_line`, `update_img`,
colorbar helpers) are structurally sound and worth preserving. Only the
high-level functions that reference the old API need rewriting.

**Alternative: Move to a separate `qdmpy-viz` package.** Premature — the
plotting module is tightly coupled to QDMpy data structures. Separation would
add complexity without benefit at this scale.

**Alternative: Switch to plotly/bokeh for interactive plots.** Out of scope for
a debt-reduction QEP. Could be a future enhancement but doesn't address the
current broken state.
