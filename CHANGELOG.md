# Changelog

All notable changes to QDMpy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed

- Converted all `logger` calls from eager f-string formatting to loguru's lazy
  `{}` placeholder syntax across 14 source files (~55 call sites). Lazy
  formatting avoids string interpolation when the log level is suppressed,
  improving performance in production (INFO+) log levels.
- Added ~25 missing log calls per the logging rule: INFO logs for
  field-processing pipeline steps, magnetic-map reconstruction, and
  `plot_qdm_display`; WARNING logs for 6 silent exception handlers in
  `plotting.py`; DEBUG logs for 13 plotting entry points; INFO for the
  brute-force D_ZFS search in `odmr/folding.py`.

### Added

- `argmin_center(data, freq)` in `fitting/guess.py` — new `@njit(parallel=True)`
  center estimator that uses the frequency of the deepest dip per pixel. Replaces
  `cumsum_center` which produced catastrophically wrong guesses (~10 MHz error)
  for pixels with strong B111 shifts where the resonance sits near the frequency
  range edge.
- `halfpower_width(data, freq)` in `fitting/guess.py` — new `@njit(parallel=True)`
  estimator that measures envelope HWHM directly from half-power points of each
  pixel spectrum, with one-sided baseline selection to handle edge-shifted dips.
  Replaces `cumsum_width` for initial width guesses.
- `scripts/benchmark_guesses.py` — benchmark script that evaluates guess quality
  via model residuals on real ODMR data, with baseline save/compare workflow.
- `scripts/plot_guess_comparison.py` — visual comparison of old vs new parameter
  guesses overlaid on raw ODMR spectra for sample pixels.

- `plot_b111_map(result, component)` in `plotting.py` — dedicated B111 component
  map with symmetric `RdBu_r` colormap and 99th-percentile color limits.
- `plot_measurement_images(measurement)` in `plotting.py` — side-by-side display
  of light and laser optical images.
- `plot_qdm_display(result, measurement=None, n_sample_pixels=3)` in `plotting.py`
  — comprehensive overview: B111 remanent/induced, chi-squared, centre/contrast/
  linewidth maps; optionally optical images and representative pixel ODMR spectra
  with fitted model curves overlaid.
- `Measurement.plot()` — shorthand for `plot_measurement_images(self)`.
- `Measurement.display(result)` — shorthand for `plot_qdm_display(result, self)`.
- `QDMResult.display(measurement=None)` — shorthand for
  `plot_qdm_display(self, measurement)`.
- `FitResult.plot('b111_remanent')` and `FitResult.plot('b111_induced')` now
  delegate to `plot_b111_map` rather than raising `ParameterError`.

- `plot_fit_result_field_map` and `plot_fit_result_overview` in `plotting.py` now
  use `result.b111_remanent` (µT, diverging colormap) for multi-range models
  (ESR14N, ESR15N) instead of `calculate_b_field()`, which raised `ParameterError`
  for those models. Single-range fits fall back to the legacy T-unit display.

### Fixed

- **Center guess catastrophically wrong for shifted pixels** — `cumsum_center`
  produced ~10 MHz errors for pixels with strong B111 fields where the resonance
  shifts to the edge of the frequency range. The cumsum normalization S-curve
  saturates and the 0.5 crossing lands on the wrong side. Replaced with
  `argmin_center` (simple `freq[argmin(spectrum)]`) which is robust to shifts
  and 4x faster.
- **Width measured envelope span, not Lorentzian HWHM** — replaced `cumsum_width`
  with `halfpower_width` (direct half-power point measurement) and added model-aware
  AHYP correction: `individual_HWHM = envelope_HWHM - AHYP`, floored at 0.3 MHz.
  This produces width guesses ~2-5x closer to true values for multi-peak models.

- **QEP-048 (root-cause H1)** — `normalize_pixel` in `fitting/guess.py` previously
  subtracted the hardcoded value `1.0` before computing the cumulative sum used for
  center and width initial guesses. This implicitly assumed max-normalization (which
  guarantees off-resonance = 1.0) and produced systematically drifted cumsum profiles
  for mean-normalized data (off-resonance ≈ 1.03–1.10), degrading fit quality. The
  function now estimates the actual baseline from the mean of the first and last 10%
  of frequency points, making the initial guesses correct and normalization-independent.

### Changed

- **QEP-048 (investigation)** — `NormalizationProcessor` reinstates `method='max'` as a
  **deprecated** option (raises `DeprecationWarning` at construction, not `ValueError`)
  to allow direct A/B comparison between max-norm and mean-norm pipelines. Max-norm
  remains physically incorrect for fluorescence correction and will be removed in a
  future release. Use `method='mean'`.

- **QEP-048 (partial)** — `NormalizationProcessor` now uses mean-normalization exclusively:
  - Default `method` changed from `'max'` to `'mean'`
  - `method='max'` raises `ValueError` at construction with a migration message explaining why max-normalization is physically invalid (it destroys per-pixel baseline variation needed for fluorescence correction)
  - `OdmrSettings.norm_method` Literal narrowed to `"mean"` only
  - Updated all examples, scripts, and integration tests to use `method='mean'`

### Added

- **Metadata TOML support** in `Measurement.from_folder()`:
  - New `load_metadata_toml()` function in `io.py` for loading TOML configuration files
  - `Measurement.__init__` now accepts optional `metadata` parameter for initialization
  - `from_folder()` automatically loads `metadata.toml` from measurement folder if present
  - Graceful error handling: missing files return empty dict, malformed TOML is skipped with warning
  - Enables users to annotate experiments (sample name, temperature, operator, notes, etc.) without programmatic configuration

- Consolidated all plotting into `plotting.py`:
  - `plot_odmr_spectra()` -- plot all ODMR spectra for a pixel (moved from `odmr/manager.py`)
  - `plot_fluorescence_correction()` -- preview fluorescence correction (moved from `odmr/processors.py`)
  - `plot_model_detection()` -- visualize auto-model detection (moved from `fitting/guess.py`)
  - `plot_magnetic_component()` -- display MagneticMap component (moved from `magnetic_map.py`)
  - Original call sites now delegate to `plotting.py`; no API changes
  - Removed all direct matplotlib imports from non-plotting modules (except `io.py` for `mpimg.imread`)

- **QEP-011** -- Spectral folding diagnostic plots:
  - `FoldedODMR.d_candidates` and `search_residual` fields store brute-force D_ZFS search landscape
  - `FoldedODMR.plot()` convenience method for quick diagnostic overview
  - `plot_folding_search_landscape()` -- D candidate vs mean residual per polarity
  - `plot_folding_mean_spectrum()` -- spatially-averaged folded and antisymmetric spectra
  - `plot_folding_overview()` -- 2x2 panel combining search landscape, D_ZFS map, and fold residual map
  - 5 new tests for search diagnostics, 5 smoke tests for plot functions

- **QEP-011** -- Spectral folding orchestration in Measurement:
  - `Measurement.fold_odmr(settings=None)` -- creates `SpectralFolder`, runs fold, caches result
  - `Measurement.folded_odmr` property -- returns cached `FoldedODMR` or raises `DataNotLoadedError`
  - `Measurement.fit_folded_odmr()` now uses cached folded data when called without arguments; explicit `folded=` arg still works for backward compat
  - `fit_folded_odmr()` now calls `_validate_fit_prerequisites()` (GPU check was missing)
  - Both `fit_odmr()` and `fit_folded_odmr()` now use `_fit_model` as the default model name
  - `FoldedODMR`, `FoldingSettings`, `SpectralFolder` imported at module level in `measurement.py`
  - Updated `notebooks/04-spectral-folding.ipynb` to use `m.fold_odmr()` / `m.fit_folded_odmr()` API
  - Fixed broken mock paths in `tests/test_measurement.py` (`QDMpy.*` -> `qdmpy_core.*`)
  - Added 12 new tests covering fold_odmr, folded_odmr property, fit_folded_odmr cached/explicit, GPU validation, and _fit_model wiring

- **QEP-046** — Notebook tutorials for three user types:
  - `src/QDMpy/testing.py` — three public helpers for tutorials and tests: `make_synthetic_odmr_data()`, `make_synthetic_fit_result()`, `make_synthetic_qdm_result()`; all exported from top-level `QDMpy`
  - `notebooks/01-quickstart.ipynb` — User 1 ("fit and be done"): `QDMpy.load()` → `fit_odmr()` → B111 maps → `magnetic_map` → save/load
  - `notebooks/02-exploration.ipynb` — User 2 ("explore the data"): raw data inspection, processing pipeline, spectrum plots, parameter maps, B111 xarray, binning iteration, export
  - `notebooks/03-extending.ipynb` — User 3 ("develop my own algorithms"): custom `Model`, custom `Processor`, custom `FieldReconstructor`, standalone `FitManager`
  - `.github/workflows/notebooks.yml` — CI notebook execution on pushes to `claude` branch with `notebooks/**` or `src/**` path filter
  - `README.md` Quick Start updated to the new one-line `QDMpy.load()` API; Notebooks table added

- **QEP-045** — Developer extension points:
  - `Model` ABC docstring enhanced with full custom-model contract and copy-paste example
  - `ModelRegistry.available_models()` — returns sorted list of all registered model names
  - `Processor` protocol (`typing.Protocol`, `@runtime_checkable`) added to `odmr/processors.py` and exported from top-level `QDMpy`; `BaseProcessor.describe()` added so all built-in processors satisfy the protocol
  - `FieldReconstructor` protocol added to `magnetic_map.py` and exported from top-level `QDMpy`; `MagneticMap.from_b111()` accepts optional `reconstructor: FieldReconstructor | None = None` parameter to bypass the default Fourier inversion
  - `QDMResult.reconstructor` field — passed through to `MagneticMap.from_b111()` during lazy `magnetic_map` build
  - `docs/extending.md` — developer guide covering all three extension points with runnable code examples
  - `tests/test_extensions.py` (16 tests) covering custom model registration, protocol conformance for `Processor` and `FieldReconstructor`, custom reconstructor round-trip through `QDMResult`, and `available_models()` listing

- **QEP-044** — Convenience methods for exploration:
  - `ODMR.spectrum(y, x, polarity='neg', freq_range='low', *, processed=True)` → `(freq_ghz, intensity)` tuple for quick single-pixel inspection
  - `ODMR.plot_spectra(y, x)` — 2×2 matplotlib grid of all (polarity × freq_range) spectra at a pixel
  - `FitResult.plot(param='center', **kwargs)` — thin wrapper over `plot_fit_result_parameter_map`
  - `FitResult.show(**kwargs)` — thin wrapper over `plot_fit_result_overview`
  - `QDMResult.plot()` / `QDMResult.show()` — delegate to `fit_result`
  - `tests/test_convenience.py` (19 tests) covering spectrum shape/values, subplot grid, and plot delegation
- **QEP-043** — `QDMpy.load()` entry-point function:
  - `Measurement.from_folder(path, *, bin_factor, model, pixel_spacing, normalize, fluorescence_correction, output_directory)` classmethod — full pipeline in one call
  - `QDMpy.load(path, **kwargs)` top-level convenience function delegating to `from_folder()`; added to `__all__`
  - `fluorescence_correction: float | None = 0.2` — applies `FluorescenceCorrectionProcessor`; `None` skips it
  - Missing light/laser images fall back to `np.zeros(scan_dimensions)` with a warning instead of raising
  - Image files discovered via `os.listdir` filtered by `'light'`/`'laser'` in filename
  - `tests/test_load.py` (18 tests) covering processor pipeline composition, image fallback, file filtering, and config pass-through
- **QEP-042** — Fixed top-level API surface (`QDMpy/__init__.py`):
  - Added user-facing entry points: `Measurement`, `QDMResult`
  - Added data loading: `MatlabLoader`, `ODMRData`, `ODMR`
  - Added processing: `BinningProcessor`, `NormalizationProcessor`, `FluorescenceCorrectionProcessor`, `OutlierProcessor`
  - Added fitting: `FitManager`, `FitResult`, `Model`, `ModelRegistry`
  - Explicit `__all__` now enumerates all 26 public names in usage-frequency order
  - `tests/test_imports.py` (25 parametrised tests) — smoke tests asserting every `__all__` entry is importable and the correct kind (class / callable)
- **QEP-041** — `QDMResult` top-level result container:
  - New `src/QDMpy/result.py` module with `QDMResult` Pydantic model
  - `Measurement.fit_odmr()` now returns `QDMResult` instead of bare `FitResult`
  - All `FitResult` properties delegated directly (`b111_remanent`, `b111_induced`, `b111`, `centers`, `chi2`, `scan_dimensions`, `pixel_spacing`, `model_name`, `metadata`, etc.)
  - `QDMResult.magnetic_map` lazily constructs `MagneticMap` (Fourier 3D field reconstruction) on first access — zero cost for users who only need B111
  - Optional `nv_axis` parameter on `QDMResult`; when `None`, settings default is used at `MagneticMap` construction time
  - `QDMResult.save(path)` / `QDMResult.load(path)` for NPZ round-trip including `nv_axis`
  - `QDMResult` exported from `QDMpy` top-level `__init__.py`
  - `tests/test_qdm_result.py` (25 tests) covering construction, delegation, lazy magnetic map, caching, and save/load round-trips

### Fixed
- **QEP-035** — Critical correctness bugs:
  - `FitResult.get_parameter_map()` now properly flattens multi-dimensional parameters before reshape (e.g. shape `(n_pol, n_frange, n_pixel)` → flat → reshaped to `(H, W)`)
  - `FitResult._compute_b_field()` now guards against multi-range models; raises clear `ParameterError` directing users to use `b111` property instead
  - `FitResult.centers` property docstring corrected: frequencies are in GHz (was incorrectly documented as Hz)
  - `ODMR.load_xarray()` now uses explicit named argument `ODMRData(data=data)` instead of positional (safer against field reordering)
- **QEP-038 M4** — `FitResult.load_results()` classmethod now returns `FitResult` instance (was returning dict); properly deserializes parameters and metadata from NPZ object arrays
- Tutorial notebook fixed: ESR14N model produces `contrast_0/1/2` (for three hyperfine dips), not single `contrast` key; updated cells to use correct parameter names

### Changed
- **QEP-036** — Resolved cross-layer imports (fitting ↔ odmr layers are peers):
  - Moved `POLARITY_LABELS`, `FRANGE_LABELS`, `validate_frequencies()` to `QDMpy.constants` (neutral ground)
  - Moved `is_pygpufit_available()` to `QDMpy.settings` (from root `__init__.py`)
  - `fitting/manager.py` and `fitting/result.py` now import from constants/settings instead of `odmr.data`
  - Updated all tests to import from new locations
- **QEP-037** — Enforced immutability:
  - `FitManager.model_name` no longer has a setter; model is fixed at construction (prevents silent constraint destruction)
  - `ODMRData` is now frozen (`ConfigDict(frozen=True)`) to prevent external mutation
  - `FitResult.parameters` arrays are write-protected (`.flags.writeable = False`) to prevent cache invalidation
- **QEP-038 M1** — Physics analysis layer separation: moved `b111_from_dip_positions()` from `odmr/data.py` to new `odmr/analysis.py` module (single responsibility)
- **QEP-038 M2** — Image I/O extraction: moved `has_csv()`, `get_image_file()`, `get_image()` from `measurement.py` to new top-level `QDMpy/io.py` module
- **QEP-038 M3** — Removed duplicate auto-model detection: deleted `Measurement._detect_model()` (was identical to `FitManager._resolve_auto_model`); `fit_odmr()` now passes model directly to FitManager

### Removed
- **QEP-035** — Dead code: removed `FitResult._calc_delta_from_multi_centers()` method and associated test class (was only triggered by non-existent multi-center model parameters)
- **QEP-038 M3** — Removed `TestDetectModel` test class (3 tests testing removed `_detect_model` method)
- **QEP-040** — Removed hardcoded magic number `0.001` from `FluorescenceCorrectionProcessor`; now uses `FLUORESCENCE_DELTA_THRESHOLD` constant in `QDMpy.constants`

### Added
- **QEP-038 M5** — Module decomposition for cleaner architecture:
  - New `src/QDMpy/fitting/constraints.py` (124 lines): `ConstraintManager` class and `CONSTRAINT_TYPES` constant
  - New `src/QDMpy/fitting/guesser.py` (128 lines): `ParameterGuesser` class with cached parameter estimation and model-specific width thresholds
  - Updated `src/QDMpy/fitting/__init__.py` to import and re-export classes from new modules
- **QEP-039** — New `tests/conftest.py` (187 lines) with shared test fixtures: `rng`, `sample_numpy_data`, `sample_frequencies`, `sample_data`, `sample_parameters`, `sample_fit_result`, plus `make_xr_data()` helper and `MOCK_SETTINGS` configuration

### Added
- **QEP-030** — `BaseProcessor` is now a Pydantic `BaseModel` with `frozen=True`; all processor
  config is declared as validated fields (e.g. `BinningProcessor.bin_factor: int = Field(gt=0)`)
- **QEP-030** — Each processor carries a `type: Literal[...]` discriminator field enabling
  discriminated-union deserialization via `ProcessorSpec` / `_adapter = TypeAdapter(ProcessorSpec)`
- **QEP-030** — `BaseProcessor.to_config()` serializes any processor to a plain JSON-compatible dict
- **QEP-030** — `ODMRProcessorManager.from_config(config)` reconstructs a full pipeline from a
  serialized config list (e.g. `processed_data.metadata['pipeline']`)
- **QEP-030** — `ODMRProcessorManager.pipeline_config` property returns the current pipeline as a
  list of config dicts
- **QEP-030** — `ODMRProcessorManager.process()` writes a `'pipeline'` key to output metadata
  containing the complete ordered list of processor configs applied

### Changed
- **QEP-030** — Processors no longer write ad-hoc keys to `ODMRData.metadata`; the manager owns
  the single canonical pipeline snapshot (`metadata['pipeline']`)
- **QEP-030** — `BinningProcessor` validation moved from manual `if bin_factor <= 0: raise` to
  Pydantic `Field(gt=0)` — raises `pydantic.ValidationError` instead of `DataValidationError`
- **QEP-030** — `ODMRProcessorManager.list_processors()` now returns `p.type` (the discriminator
  string) rather than `p.__class__.__name__`

### Changed
- **QEP-025** — Semantic coordinate labels: `pol_0`/`pol_1` → `neg`/`pos`, `frange_0`/`frange_1` → `low`/`high`
  in both `ODMRData.from_numpy` and `MatlabLoader`; labels exported as `POLARITY_LABELS`/`FRANGE_LABELS`
  constants from `QDMpy.odmr.data`
- **QEP-025** — `delta_resonance` tensor shape `(n_pol, 2, H, W)` → `(n_pol, H, W)` `xr.DataArray`
  with `polarity` coordinate; sign applied per polarity (neg=-1, pos=+1) — eliminates the ambiguous
  ±sign axis that previously caused the B111 bug
- **QEP-025** — `FitResult.b111` returns `xr.Dataset` with `'remanent'` and `'induced'` DataArrays
  (units='µT'); `b111_remanent` and `b111_induced` properties kept as `.values` shims

### Performance
- **QEP-022** — Vectorized `esr14n`, `esr15n`, `esrsingle` model functions: replaced
  Python `for p in parameter` loops with numpy broadcasting over `(N, 1)` × `(n_freq,)`
  arrays. Benchmarked **28–33× speedup** at 9k pixels (bin=2) vs the loop implementation.

### Changed
- **QEP-029** — `FitManager.__init__` no longer accepts `data` or `frequencies`; configuration
  only (model, constraints, settings). Call `fit_manager.fit(data, frequencies)` to run fitting
  and receive a `FitResult` directly. Same `FitManager` instance can be reused across calls.
- **QEP-029** — Auto model detection deferred to first `fit()` call when `model_name='auto'`
- **QEP-029** — `Measurement.fit_odmr()` delegates entirely to `FitManager.fit()`;
  removed `_extract_fit_parameters()` and `_compute_quality_metrics()` private helpers
- **QEP-029** — `reshape_results` / `reshape_result` replaced by `_reshape_frange_results(raw, data_shape)`
  taking explicit `data_shape` arg — no more `_current_data_shape` instance variable

### Removed
- **QEP-029** — Removed `FitManager.fit_odmr()`, `data` property/setter, `_reset_fit()`,
  `parameter` property, `get_param()`, `initial_parameter` property, `get_initial_parameter()`,
  `fitted` property, `_flat_data` property, `_current_data_shape` instance variable
- **QEP-029** — Removed `Measurement._extract_fit_parameters()` and
  `Measurement._compute_quality_metrics()` static methods (quality metrics now in `FitManager.fit()`)

### Removed
- **QEP-032** — Deleted 7 backward-compatibility shim files (`fit.py`, `result.py`, `models.py`,
  `guess.py`, `io.py`, `odmr/odmr.py`, `odmr/validation.py`); use canonical paths in `fitting/`
  and `odmr/manager.py` instead
- **QEP-032** — Removed `Model.parameters_unique` and `FitManager.model_params_unique` property
  aliases; use `parameter_names` on both classes
- **QEP-032** — Removed deprecated exception aliases `CantImportError` and `WrongFileNumberError`;
  use `DependencyError` and `DataValidationError` respectively
- **QEP-032** — Removed `ModelRegistry._initialize_constraints()` (duplicated `ConstraintManager`)
- **QEP-032** — Removed `process` and `info` CLI subcommands (were raising `NotImplementedError`)
- **QEP-032** — Removed `__main__` block with hardcoded path from `measurement.py`
- **QEP-032** — Removed unused `Measurement._B111` attribute
- **QEP-032** — Removed `visualize_fluorescence_correction` alias in `odmr/processors.py`

### Changed
- **QEP-032** — `ModelRegistry.register()` now reads `model_cls.name` (ClassVar) instead of
  instantiating a throwaway instance; concrete model classes declare `name: ClassVar[str]`
- **QEP-032** — `OutlierProcessor`: renamed `threshold` → `z_score_threshold`, updated default
  from `0.001` to `0.003`, removed hidden `* 3` internal multiplier
- **QEP-032** — `FluorescenceCorrectionProcessor.process()` now matches the `BaseProcessor`
  interface (no extra kwargs); configure `correction_factor` at construction time instead
- **QEP-032** — `models` CLI command now prints model names, peak counts, and parameter lists;
  `--detailed` flag shows per-parameter units

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
