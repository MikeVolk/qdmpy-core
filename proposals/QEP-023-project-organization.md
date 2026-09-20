# QEP-023: Project Organization and Naming Cleanup

**Status:** Implemented (commit 3923ca6, 2026-02-20)
**Priority:** Medium
**Affects:** Package-wide

## Overview

The package has grown organically through the refactoring QEPs, and several
organizational patterns have drifted. This QEP addresses module placement,
naming consistency, responsibility boundaries, and the top-level `__init__.py`
overload.

---

## 1. Fitting modules are scattered across the top level

### Problem

Four tightly-coupled modules sit at the package root alongside unrelated code:

```
src/QDMpy/
├── fit.py          # FitManager, ConstraintManager, ParameterGuesser
├── result.py       # FitResult
├── guess.py        # guess_model, guess_center, guess_width, ...
├── models.py       # Model, ModelRegistry, ESR14N/15N/SINGLE
├── measurement.py  # Measurement (integrator)
├── plotting.py     # 600 lines of matplotlib helpers
├── io.py           # 3 image-loading functions
├── utils.py        # grab bag
├── constants.py    # physics + algorithm defaults mixed
├── settings.py     # ok
├── exceptions.py   # ok
└── odmr/           # well-organized subpackage
```

The dependency graph makes the cluster obvious:

```
fit.py  →  models.py, guess.py, constants.py
result.py  →  constants.py
guess.py  →  models.py, constants.py
models.py  →  constants.py
```

These four modules form a cohesive "fitting" domain. They have no dependency on
`odmr/`, `measurement.py`, `plotting.py`, or `io.py`. Keeping them flat at the
root mixes concerns and makes the package harder to navigate.

### Proposed fix

Create a `fitting/` subpackage:

```
src/QDMpy/
├── fitting/
│   ├── __init__.py       # exports: FitManager, FitResult, Model, ModelRegistry
│   ├── manager.py        # FitManager, ConstraintManager, ParameterGuesser (was fit.py)
│   ├── result.py         # FitResult
│   ├── guess.py          # guess_model, guess_center, ...
│   └── models.py         # Model, ModelRegistry, ESR14N/15N/SINGLE
├── odmr/                 # unchanged
├── cli/                  # unchanged
├── measurement.py
├── plotting.py
├── io.py
├── constants.py
├── settings.py
├── exceptions.py
└── utils.py
```

Rename `fit.py` → `fitting/manager.py` to avoid the `from QDMpy.fitting.fitting`
stutter (like `odmr/odmr.py` already suffers from — see issue 5 below).

---

## 2. `__init__.py` has too many responsibilities

### Problem

The package `__init__.py` (136 lines) manages:

- Version string
- Four path constants (`PROJECT_PATH`, `CONFIG_PATH`, `CONFIG_FILE`, `DESKTOP`)
- Config file creation/reset (2 functions)
- Logging configuration (1 function)
- Settings singleton (2 functions + module global)
- GPU availability check (1 function)
- Matplotlib rcParams
- Module imports (`from . import io`)
- Test data location (1 function)

This is an initialization dumping ground. Several items don't belong:

| Item | Problem |
|------|---------|
| `DESKTOP = Path().home() / "Desktop"` | Unused anywhere in codebase, platform-specific |
| `PROJECT_PATH` | Unused anywhere in codebase |
| `from . import io` | `io.py` is never imported via `QDMpy.io` by any module |
| `test_data_location()` | Test infrastructure, not package API |
| `mpl.rcParams["figure.facecolor"] = "white"` | Side effect on import |

### Proposed fix

- Delete `DESKTOP` and `PROJECT_PATH` (unused).
- Delete `from . import io` (unused import).
- Move `test_data_location()` to `tests/conftest.py` as a fixture.
- Move matplotlib rcParams configuration into `plotting.py` (only relevant when
  plotting).
- Keep settings singleton and logging setup in `__init__.py` — that's appropriate
  for package-level initialization.

---

## 3. `constants.py` mixes physics with algorithm tuning parameters

### Problem

```python
# Physical constants
GAMMA_NV = 28.024      # GHz/T
D_ZFS = 2.870          # GHz
AHYP_14N = 0.002158    # GHz
AHYP_15N = 0.0015      # GHz

# Algorithm tuning — NOT physical constants
DEFAULT_VMIN = 0.3
DEFAULT_VMAX = 0.7
PROMINENCE = 0.0004
```

`DEFAULT_VMIN`, `DEFAULT_VMAX`, and `PROMINENCE` are algorithm parameters for
`guess.py`. They're not universal constants — they're defaults that could change
between datasets. Mixing them with actual physical constants obscures the
distinction.

### Proposed fix

- Keep physics constants in `constants.py`: `GAMMA_NV`, `D_ZFS`, `AHYP_14N`,
  `AHYP_15N`, conversion factors.
- Move algorithm defaults into `settings.py` as part of a `GuessSettings` model,
  or define them as module-level defaults in `guess.py` where they're used.

---

## 4. Confusing property and parameter names on `Model`

### Problem

```python
class Model(ABC):
    parameters_unique: list[str]     # e.g. ['center', 'width', 'contrast_0', 'contrast_1', 'offset']

    @property
    def parameter(self) -> list[str]:  # returns ['center', 'width', 'contrast', 'contrast', 'offset']
        return [self.parameter_types[p] for p in self.parameters_unique]
```

Issues:

| Name | Returns | Confusion |
|------|---------|-----------|
| `parameters_unique` | The actual parameter names | "unique" compared to what? They're just the parameter names |
| `parameter` (singular) | Parameter *type* categories | Looks like it should return one parameter; actually returns a list of types |
| `parameter_types` | Dict mapping name → type | Fine |
| `n_parameters` | Count | Fine |

Then `FitManager` re-exports these with yet more names:

```python
@property
def model_params(self) -> list[str]:        # delegates to model.parameter (types!)
    return self._model.parameter

@property
def model_params_unique(self) -> list[str]: # delegates to model.parameters_unique
    return self._model.parameters_unique
```

So `FitManager.model_params` returns parameter *types* (not params), and
`model_params_unique` returns actual parameter *names*. The naming is
inverted from what a reader would expect.

### Proposed fix

Rename on `Model`:

| Old | New | Returns |
|-----|-----|---------|
| `parameters_unique` | `parameter_names` | `['center', 'width', 'contrast_0', ...]` |
| `parameter` | `parameter_types_list` (or remove) | `['center', 'width', 'contrast', ...]` |

The `parameter` property is only used in one place (`FitManager._param_idx`)
and could be replaced by a direct lookup into `parameter_types`. Consider
removing it entirely.

On `FitManager`, either delegate with matching names or drop the delegation
properties and have callers use `fit_manager.model.parameter_names` directly.

---

## 5. `odmr/odmr.py` module/class name stutter

### Problem

```python
from QDMpy.odmr.odmr import ODMR
```

The `odmr` directory and `odmr.py` module share a name, forcing the stuttered
import. This already exists and is awkward.

### Proposed fix

Rename `odmr/odmr.py` → `odmr/manager.py`:

```python
from QDMpy.odmr.manager import ODMR
```

The `odmr/__init__.py` already re-exports `ODMR`, so most users import from
`QDMpy.odmr` and won't notice the internal rename.

---

## 6. `io.py` (top-level) is misplaced and nearly empty

### Problem

Top-level `io.py` contains exactly 3 functions:
- `has_csv(lst)` — checks if a list has CSV files
- `get_image_file(lst)` — picks first CSV or JPG from a list
- `get_image(folder, lst)` — loads an image from folder

These are only relevant to `Measurement`, which loads light/laser reference
images. No other module imports `QDMpy.io` (the `from . import io` in
`__init__.py` is itself unused).

Meanwhile `odmr/io.py` handles MATLAB data loading — a completely separate
concern using the same module name. Having two `io.py` files at different
levels is confusing.

### Proposed fix

Option A: Fold the 3 functions into `measurement.py` as private helpers
(they're only ~40 lines of logic).

Option B: Rename to `images.py` to distinguish from `odmr/io.py`.

---

## 7. `utils.py` is a classic grab bag

### Problem

`utils.py` contains 6 unrelated functions:

| Function | Domain | Used by |
|----------|--------|---------|
| `millify` | String formatting | plotting (indirectly) |
| `idx2rc` | Coordinate transform | unknown |
| `rc2idx` | Coordinate transform | unknown |
| `polyfit2d` | 2D polynomial fitting | unknown |
| `rms` | Statistics | unknown |
| `double_norm` | Array normalization | `plotting.py` |

Plus a dead `main()` function.

### Proposed fix

Check actual usage. Functions used only by one module should move into that
module as private helpers. Functions used by multiple modules can stay, but
`utils.py` should not be a default destination for new code. Consider renaming
to `math_utils.py` if the remaining functions are all mathematical.

---

## 8. `guess_initial_fit_parameters()` duplicates `ParameterGuesser.guess()`

### Problem

`guess.py` has both:
- `guess_initial_fit_parameters(data, freq, model)` — standalone function
- Used by nobody in the codebase

And `fit.py` has:
- `ParameterGuesser.guess(flat_data)` — class method doing the same thing

Both iterate over `model.parameters_unique`, dispatch to `guess_center()`,
`guess_contrast()`, `guess_width()`, and stack results. One is dead code.

### Proposed fix

Delete `guess_initial_fit_parameters()` from `guess.py`. The
`ParameterGuesser` class in `fit.py` is the canonical path.

---

## 9. `plotting.py` mixes two audiences

### Problem

`plotting.py` (617 lines) contains two kinds of functions:

1. **High-level result plots** (lines 32-223): `plot_fit_result_field_map`,
   `plot_fit_result_parameter_map`, `plot_fit_result_overview` — these take a
   `FitResult` and produce complete figures. These are the public API.

2. **Low-level widget helpers** (lines 226-617): `plot_light_img`,
   `update_line`, `update_marker`, `update_img`, `toggle_img`, `update_clim`,
   `update_cbar`, `detect_extent`, `get_vmin_vmax`, `get_color_norm`,
   `plot_overlay`, `plot_outlier`, `plot_quality_data`, `plot_data`,
   `plot_laser_img`, `plot_fluorescence` — these are GUI widget internals
   from the old PySide6 app. They operate on raw `AxesImage` objects and
   have nothing to do with the current library API.

The low-level helpers are orphaned: the GUI that used them doesn't exist in the
new codebase.

### Proposed fix

- Keep the 3 high-level `plot_fit_result_*` functions.
- Audit the low-level helpers for actual usage. If nothing in the current
  codebase calls them, delete them. If a future GUI is planned, they can be
  re-added when needed (YAGNI).

---

## 10. `odmr/validation.py` is too thin to justify a module

### Problem

`odmr/validation.py` contains a single function: `validate_frequencies()` (28
lines of logic). It's imported in two places: `odmr/data.py` and `fit.py`.

A one-function module adds navigational overhead. The function is a data
validation concern that belongs with the data it validates.

### Proposed fix

Move `validate_frequencies()` into `odmr/data.py` (where most of its callers
are). Delete `odmr/validation.py`.

---

## Summary of changes

| Issue | Action | Impact |
|-------|--------|--------|
| Scattered fitting modules | Create `fitting/` subpackage | Moderate (import paths change) |
| `__init__.py` overload | Delete dead code, move test helpers | Small |
| Constants mixing | Move algorithm defaults to settings/guess | Small |
| Model naming | Rename `parameters_unique`→`parameter_names`, drop `parameter` | Moderate |
| `odmr/odmr.py` stutter | Rename to `odmr/manager.py` | Small (re-exported) |
| Top-level `io.py` | Fold into `measurement.py` or rename | Small |
| `utils.py` grab bag | Audit usage, inline or rename | Small |
| Duplicate guess function | Delete `guess_initial_fit_parameters` | Small |
| `plotting.py` bloat | Delete orphaned GUI helpers | Small |
| Thin validation module | Merge into `odmr/data.py` | Small |

## Migration

All import path changes should be accompanied by re-exports from the old
locations for one release cycle, then removed. Since the package is pre-1.0
(version 0.1.0a), breaking changes are acceptable if documented.
