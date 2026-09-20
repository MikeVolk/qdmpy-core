# QEP-010: CLI and Plotting Coverage

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P4 |
| **Complexity** | L |
| **Depends on** | Nothing (QEP-001 superseded by QEP-011) |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |
| **Revised** | 2026-03-22 | Marked Implemented -- all spec items completed via organic development |

## QEP-011 Impact

QEP-011 (now Implemented) removed ~40 lines of broken code from the old
`plotting.py` and deleted `tests/test_odmr.py` (257 lines). The plotting module
was subsequently refactored into a package (`qdmpy.plotting`) with submodules
(`_common`, `display`, `fields`, `fit`, `odmr`). The plotting logic for
fluorescence correction now lives in `qdmpy.plotting.odmr.plot_fluorescence_correction`,
with a thin delegate retained in `processors.py` for backwards compatibility.
CLI and plotting tests have been added.

## Motivation

Two modules significantly drag down test coverage:

| Module | Coverage | Lines | Uncovered |
|--------|----------|-------|-----------|
| `cli/qdmpy_cli.py` | 0% | ~120 | ~120 |
| `plotting.py` | 46% | ~400 | ~216 |

Together they account for ~336 uncovered lines, blocking the 80% coverage target.

Additionally:
- `plotting.py` has broken imports referencing `QDMpy._core` which doesn't exist
  in the new codebase
- `preview_fluorescence_correction()` (96 lines) is in `processors.py` where it
  doesn't belong — it's a plotting function mixed into a data processing module
- Test for `plotting.py` skips with a comment about `QDMpy._core`

## GUI Integration Requirements

No GUI impact. The CLI and plotting modules are core-only concerns:

- `qdmpy.cli` is a standalone command-line entry point not used by qdmpy-gui.
- `qdmpy.plotting` provides matplotlib-based plotting for notebook/script use.
  The GUI uses pyqtgraph for its own rendering and does not import from
  `qdmpy.plotting`.
- No settings keys, data contracts, or API signatures consumed by the GUI were
  changed.

## Specification

### 1. Fix plotting.py Imports

Identify and fix all references to `QDMpy._core`:

```python
# Before (broken)
from QDMpy._core import SomeClass

# After (use correct module paths)
from QDMpy.odmr.data import ODMRData
from QDMpy.result import FitResult
```

Determine what `_core` was in the old codebase and map those imports to the new
module structure.

### 2. Move preview_fluorescence_correction

Move `preview_fluorescence_correction()` from `processors.py` to `plotting.py`:

```python
# src/QDMpy/odmr/processors.py
# REMOVE: def preview_fluorescence_correction(...)  # 96 lines

# src/QDMpy/plotting.py
# ADD: def preview_fluorescence_correction(...)
```

Update any imports that reference the old location. This is a pure SRP fix —
plotting belongs in the plotting module.

### 3. Add CLI Tests

Create `tests/test_cli.py` with tests for the CLI entry points:

```python
import pytest
from click.testing import CliRunner
from QDMpy.cli.qdmpy_cli import main  # or whatever the entry point is


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Usage' in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0

    def test_process_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ['process', 'nonexistent.qdmpy'])
        assert result.exit_code != 0

    # ... test each subcommand
```

### 4. Add Plotting Tests

Create/fix `tests/test_plotting.py` with matplotlib Agg backend:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

import pytest
import numpy as np
import matplotlib.pyplot as plt
from QDMpy.plotting import (
    plot_odmr_spectrum,
    plot_fit_result,
    plot_magnetic_field_map,
    preview_fluorescence_correction,
)


class TestPlotODMRSpectrum:
    def test_basic_plot(self, sample_data):
        fig, ax = plot_odmr_spectrum(sample_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_fit(self, sample_data, sample_fit):
        fig, ax = plot_odmr_spectrum(sample_data, fit=sample_fit)
        assert len(ax.lines) >= 2  # data + fit
        plt.close(fig)


class TestPreviewFluorescenceCorrection:
    def test_basic(self, sample_processor_data):
        fig = preview_fluorescence_correction(sample_processor_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
```

### 5. Ensure Test Isolation

All plotting tests must:
- Use `matplotlib.use('Agg')` to avoid display requirements
- Close all figures after each test to prevent memory leaks
- Not depend on specific rendering (test structure, not pixels)
- Use `@pytest.fixture(autouse=True)` to close figures:

```python
@pytest.fixture(autouse=True)
def cleanup_plots():
    yield
    plt.close('all')
```

## Files Affected

- `src/QDMpy/plotting.py` (fix imports, add `preview_fluorescence_correction`)
- `src/QDMpy/odmr/processors.py` (remove `preview_fluorescence_correction`)
- New: `tests/test_cli.py`
- `tests/test_plotting.py` (fix/rewrite)

## Backwards Compatibility

- `preview_fluorescence_correction` moves from `processors.py` to `plotting.py`.
  Any code importing from `processors` will need updating. Since this is an
  internal API during overhaul, this is acceptable.
- No other public API changes.

## Verification

```bash
uv run pytest tests/test_cli.py tests/test_plotting.py -v
uv run pytest --cov=QDMpy --cov-report=term-missing
# Verify coverage improvement:
# cli/qdmpy_cli.py: 0% → 70%+
# plotting.py: 46% → 80%+
```

## Implementation Notes

The spec items were completed organically during the broader codebase overhaul
rather than as a single focused effort. Key differences from the original spec:

1. **Fix plotting imports** -- resolved by refactoring `plotting.py` into a
   package (`qdmpy.plotting/`) with submodules: `_common`, `display`, `fields`,
   `fit`, `odmr`. All broken `QDMpy._core` imports are gone.
2. **Move `preview_fluorescence_correction`** -- the 96-line plotting logic was
   moved to `qdmpy.plotting.odmr.plot_fluorescence_correction`. A 3-line thin
   delegate remains in `processors.py` for import compatibility.
3. **CLI tests** -- `tests/test_cli.py` (137 lines) covers parser creation,
   command dispatch, models subcommand, main entry point, and error handling.
   Uses `argparse` (not `click` as the spec assumed).
4. **Plotting tests** -- `tests/test_plotting.py` (707 lines, 50 tests) covers
   fit result plots, folding diagnostics, ODMR spectra, fluorescence correction,
   model detection, magnetic components, display layout, and B111 maps.
5. **Package name** -- the codebase uses `qdmpy` (lowercase), not `QDMpy` as
   written in the original spec examples.

## Rejection Alternatives

**Alternative: Exclude CLI and plotting from coverage targets.** Rejected because
these modules contain real logic (argument parsing, data transformation for
visualization) that can and should be tested.

**Alternative: Use snapshot testing for plots.** Deferred. Snapshot testing
(comparing rendered images) is fragile across platforms and matplotlib versions.
Structural testing (check axes, labels, data) is more robust for now.
