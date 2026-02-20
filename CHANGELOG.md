# Changelog

All notable changes to QDMpy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Performance
- **QEP-024** — `fitting/guess.py`: upgrade `cumsum_contrast`, `cumsum_center`, `cumsum_width`
  from nested `@njit(parallel=True)` loops (prange only over `n_pixel`) to a single flat
  `prange(n_pol * n_frange * n_pixel)`, exposing all pixels across all polarities and frequency
  ranges to the thread pool simultaneously. Benchmarked **2.7× speedup** at 9k pixels (bin=2)
  against the old code; gain increases with dataset size.
  Removed dead code: `guess_contrast_pixel`, `guess_center_pixel`, `guess_width_pixel`,
  `_guess_all_pixels`, `guess_initial_fit_parameters`.

### Changed
- **QEP-024** — renamed `guess_contrast/center/width` → `cumsum_contrast/center/width` to make
  the algorithm explicit; when alternative strategies are added (e.g. `fft_center`) the naming
  convention is immediately clear.

---

## 2026-02-19

### Added
- **QEP-023** — Project organisation and naming cleanup:
  - New `fitting/` subpackage: canonical home for `manager.py` (was `fit.py`), `result.py`, `guess.py`, `models.py`; public API exposed via `QDMpy.fitting.__init__`
  - `odmr/manager.py` (was `odmr/odmr.py`); merged `odmr/validation.py` into `odmr/data.py`
  - `io.py` functions (`has_csv`, `get_image_file`, `get_image`) moved to `measurement.py` as module-level helpers

### Changed
- **QEP-023** — `Model.parameter_names` replaces `parameters_unique` as the canonical attribute; `FitManager.parameter_names` replaces `model_params_unique`
- **QEP-023** — `Model.frequency_parameters` now returns only `['center']`; width is dimensionless (a.u.), not a frequency axis
- Removed orphaned GUI helper functions from `plotting.py` (~400 lines); removed dead `main()` from `utils.py`
- Removed `DESKTOP`, `PROJECT_PATH`, `test_data_location()`, and `from . import io` from `QDMpy.__init__`

### Fixed
- Backward-compatibility shims at all old module paths (`QDMpy.fit`, `QDMpy.guess`, `QDMpy.models`, `QDMpy.result`, `QDMpy.io`, `QDMpy.odmr.odmr`, `QDMpy.odmr.validation`) ensure existing code continues to work
- Property aliases `parameters_unique`, `parameter`, `model_params_unique`, `model_params` preserved on `Model` and `FitManager`

---

## 2026-02-18

### Fixed
- `FitResult._compute_b111`: corrected B111 field calculation — `b111_remanent` was always 0 and `b111_induced` had wrong sign due to incorrect axis interpretation of the `(n_pol, 2, H, W)` delta_resonance tensor; now correctly extracts `delta_res[0, 0]` (pol_0, neg-signed negDiff) and `delta_res[-1, 1]` (pol_1, pos-signed posDiff) to match QDMlab/old-QDMpy conventions

---

## 2026-02-17

### Added
- **QEP-017** — Improved loguru logging across `io.py` and `odmr/io.py`; all load/save operations emit structured log messages
- **QEP-007** — Pydantic data validation layer: `ODMRData` is now a `BaseModel`; `xr.DataArray` validated at construction (dims, dtype, `freq_ghz` coord required)
- **QEP-009** — Domain exception hierarchy (`QDMpyError` → `DataError` / `FittingError` / `ConfigurationError` / `DependencyError`) replacing bare exceptions across all modules
- `/memory/` folder with LLM-readable module descriptions and mermaid data-flow diagrams

### Fixed
- **QEP-015** — Resolved all 30 non-TRY003 ruff violations in core package

---

## 2026-02-16

### Added
- **QEP-004A** — Decomposed `_compute_delta_resonance` into focused private methods (`_normalize_resonance_shape`, `_calc_delta_from_single_center`, `_calc_delta_from_multi_centers`)
- **QEP-004B** — Decomposed `Measurement.fit_odmr` into `_detect_model`, `_validate_fit_prerequisites`, `_extract_fit_parameters`, `_compute_quality_metrics`
- **QEP-004C** — Extracted `ParameterGuesser` class from `FitManager`; caches initial params with `reset()` invalidation
- **QEP-006** — Dependency injection for `FitManager`: optional `settings` and `gpu_available` arguments for testability
- **QEP-005** — Self-describing models: `Model.parameter_types`, `Model.frequency_parameters`, `Model.units`; `ConstraintManager` initialised from model metadata

### Fixed
- Resolved all 41 `ty` type-checking diagnostics across 8 source files
- Stripped dead functions and F821 undefined-name errors in `plotting.py`

---

## 2026-02-15

### Added
- **QEP-002** — Eliminated global state and `sys.path` hacks; settings loaded via `get_settings()` singleton
- **QEP-003** — Unified unit system: all internal frequency values in GHz; Hz↔GHz conversion only at `odmr/io.py` input boundary and `fit.py` pygpufit boundary
- **QEP-011** — `xr.DataArray` as primary ODMR data container with named dims `(polarity, freq_range, y, x, freq_idx)` and `freq_ghz` coord
- Pydantic-settings `QDMpySettings`: TOML file + `QDMPY_*` env var support
- loguru migration: replaced all stdlib `logging` calls
- `FitResult` as standalone Pydantic model (data-only, no `FitManager` reference)
- B111 magnetic field calculations in `FitResult` (`b111_remanent`, `b111_induced`, `delta_resonance`)
- `FitResult.save_results` / `load_results` (NPZ)

### Fixed
- All ruff line-length, import-order, and style violations
- Pre-existing test collection errors

---

## 2025-06-08

### Added
- Major architecture refactor: clean separation of `ODMR`, `ODMRData`, `Measurement`, `FitManager`, `FitResult`
- `ODMRProcessorManager` with composable `BaseProcessor` pipeline (`NormalizationProcessor`, `BinningProcessor`, `OutlierProcessor`, `FluorescenceCorrectionProcessor`)
- Comprehensive test suite with integration tests (38+ tests)

### Performance
- Restored `numba` parallel processing (`prange`) in `guess_center`, `guess_contrast`, `guess_width` — up to 111× speedup on large images

### Fixed
- Type safety, dead code removal, ruff formatting

---

## 2025-06-07

### Added
- Mermaid architecture diagrams in docs
- 15N ODMR data processing sample script
- Revised tutorial focused on QDMpy public API

### Fixed
- Duplicate logging handler in package `__init__.py`
- Import errors in tutorial notebooks

---

## 2025-03-30

### Added
- `ConstraintManager` class extracted from `FitManager`
- mypy integration
- Autogenerated documentation site

### Fixed
- Tests updated for new `ConstraintManager` API
- `ruff.toml` configuration
