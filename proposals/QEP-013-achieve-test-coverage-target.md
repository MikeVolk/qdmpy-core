# QEP-013: Achieve 80% Test Coverage on Core Package

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P1 |
| **Complexity** | L |
| **Depends on** | None |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-16 |
| **Implemented** | 2026-02-19 |

## Motivation

The project enforces an 80% coverage floor (`fail_under = 80` in pyproject.toml),
but actual coverage was **26%** at the time this QEP was written. Every CI run
failed the coverage gate.

This QEP targeted 80% coverage on the **core package** (everything except
`plotting.py`, which is deferred to QEP-012). Plotting is excluded from the
coverage measurement until it is rewritten.

### Coverage before implementation

| File | Coverage | Uncovered Lines |
|------|----------|-----------------|
| `cli/__init__.py` | 0% | 26 |
| `cli/qdmpy_cli.py` | 0% | 115 |
| `cli/calculate_QDMio.py` | 0% | 55 |
| `odmr/io.py` | 43% | 40 |
| `__init__.py` | 51% | 26 |
| `guess.py` | 65% | 34 |
| `odmr/processors.py` | 75% | 37 |
| `result.py` | 81% | 39 |
| `measurement.py` | 83% | 20 |
| `fit.py` | 90% | 27 |
| `models.py` | 99% | 1 |
| `constants.py` | 100% | 0 |
| `exceptions.py` | 100% | 0 |
| `settings.py` | 100% | 0 |
| `odmr/__init__.py` | 100% | 0 |
| `odmr/data.py` | 98% | 1 |
| `odmr/odmr.py` | 91% | 5 |
| `io.py` | 93% | 2 |
| `utils.py` | 95% | 2 |

Additionally:
- **53 tests skipped** (integration tests need test data, GPU tests need pygpufit)
- **2 tests failing** (plotting kwargs tests — deferred to QEP-012)

### Coverage after implementation (2026-02-19)

**Total: 93.61%** — significantly above the 80% gate.

| File | Coverage |
|------|----------|
| `cli/__init__.py` | 100% |
| `cli/qdmpy_cli.py` | 100% |
| `__init__.py` | 97% |
| `guess.py` | 100% (numba JIT bodies excluded via `# pragma: no cover`) |
| `odmr/io.py` | 97% |
| `result.py` | 94% |
| `fit.py` | 90% |
| `constants.py` | 100% |
| `exceptions.py` | 100% |
| `settings.py` | 100% |
| `utils.py` | 95% |

Remaining skips (33): integration tests (no test data), GPU tests (requires pyGpufit),
one plotting test referencing removed `_core` module — all intentional and marked.


## Specification

### Phase 1: Exclude Plotting and Fix the Coverage Gate

#### 1a. Exclude `plotting.py` from coverage measurement

Add to `pyproject.toml` under `[tool.coverage.run]`:

```toml
omit = ["src/QDMpy/plotting.py"]
```

This immediately makes the coverage number reflect the core package health.
Plotting coverage is tracked separately under QEP-012.

#### 1b. Set a realistic intermediate `fail_under`

Temporarily lower `fail_under` to the current core-only coverage (expected
~55-60% without plotting), then ratchet up as each phase completes:
- Phase 1: set to 55% (gate goes green)
- Phase 2: set to 70%
- Phase 3: set to 80%

### Phase 2: Fix Skipped Tests and Create Core Test Fixtures

#### 2a. Create lightweight synthetic test data fixtures

The integration tests all skip because `tests/data/FOV18x/` doesn't exist.
Create a `conftest.py` fixture that generates synthetic ODMR data:

```python
@pytest.fixture
def synthetic_odmr_data():
    """Generate a small synthetic ODMR dataset with known parameters.

    Returns (2, 2, 32, 32, 10) xarray DataArray:
    - 2 polarities, 2 frequency ranges, 32x32 pixels, 10 frequencies
    - Known ESR14N dip at 2.87 GHz with 3 hyperfine peaks
    """
    ...
```

This is deterministic, fast, and doesn't require shipping data files.

#### 2b. Handle pygpufit-dependent tests gracefully

The 2 `test_fit.py` tests that require pyGpufit should:
- Be marked with `@pytest.mark.gpu`
- Have CPU-path equivalents that test the same logic using scipy fallback
- The skip message should be informative: "Requires pyGpufit (GPU fitting)"

#### 2c. Create synthetic `.mat` file fixture for `odmr/io.py`

The uncovered lines (77-135) are the MATLAB file loading path:

```python
@pytest.fixture
def synthetic_mat_file(tmp_path):
    """Create a minimal .mat file for testing."""
    data = {
        'imgStack': np.random.rand(2, 10, 32, 32),
        'freqList': np.linspace(2.8, 2.9, 10),
    }
    path = tmp_path / 'test.mat'
    scipy.io.savemat(str(path), data)
    return path
```

#### 2d. Mark integration tests with proper markers

```python
@pytest.mark.integration
@pytest.mark.skipif(not TEST_DATA_PATH.exists(), reason="Test data not available")
```

### Phase 3: Close Coverage Gaps in Core Modules

Priority order by impact (lines uncovered x module importance):

#### 3a. `cli/` — 0% coverage (196 lines)

Create `tests/test_cli.py`:

```python
def test_main_help(capsys):
    """Test CLI --help doesn't crash."""
    with pytest.raises(SystemExit, match='0'):
        main(['--help'])

def test_process_command_missing_input(capsys):
    """Test error handling for missing input file."""
    ...

def test_models_command(capsys):
    """Test listing available models."""
    ...
```

Test the CLI through the Python entry point, not subprocess calls. This gives
better coverage tracking and faster execution.

#### 3b. `guess.py` — 65% coverage (34 lines uncovered)

The uncovered lines are numba-jitted helper functions. These can be tested
directly since numba functions are callable from Python:

```python
def test_guess_center_finds_dip():
    """Test that guess_center finds the frequency of minimum intensity."""
    freqs = np.linspace(2.8, 2.9, 50)
    spectrum = 1.0 - 0.05 * np.exp(-((freqs - 2.87) ** 2) / 0.001)
    center = guess_center(spectrum, freqs)
    assert abs(center - 2.87) < 0.005
```

#### 3c. `odmr/processors.py` — 75% coverage (37 lines uncovered)

The uncovered code (L235-288) is `analyze_fluorescence_effects()` and
`preview_fluorescence_correction()`. Need test data with known fluorescence
characteristics. Note: `preview_fluorescence_correction()` is a plotting
function misplaced in processors.py — move it out and exclude.

#### 3d. `__init__.py` — 51% coverage (26 lines uncovered)

Uncovered: `make_configfile()`, `reset_config()`, `test_data_location()`.
These are utility functions testable with `tmp_path`:

```python
def test_make_configfile(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    make_configfile()
    assert (tmp_path / 'QDMpy').is_dir()
```

#### 3e. `result.py` — 81% (39 lines uncovered)

Uncovered: `_compute_b111()` shape branches and `save_results()`/`load_results()`.
The shape branching is tested indirectly but not all branches are hit — add
parametrized tests for 3D and 4D delta_resonance shapes.

#### 3f. `fit.py` — 90% (27 lines uncovered)

Uncovered: GPU fitting path (`fit_frange`), result stacking for multi-range fits.
The GPU path needs the `@pytest.mark.gpu` tests from Phase 2. The stacking logic
can be tested with mocked pygpufit returns.

### Phase 4: Exclude Intentionally Untestable Code

Add `# pragma: no cover` to:
- Platform-specific GPU detection (`is_pygpufit_available()`)
- `if __name__ == '__main__'` blocks
- Interactive matplotlib event handlers (if any remain in core)

## Files Affected

- `pyproject.toml` — exclude plotting from coverage, adjust `fail_under`
- `tests/conftest.py` — add synthetic data fixtures
- `tests/test_cli.py` — new file
- `tests/test_init.py` — expand coverage
- `tests/test_guess.py` — add numba function tests
- `tests/odmr/test_io.py` — add synthetic mat file tests
- `tests/odmr/test_processors.py` — add fluorescence analysis tests
- `tests/test_result.py` — add shape branch coverage
- `tests/test_fit.py` — add mocked GPU path tests
- `tests/integration/` — add synthetic data support and proper markers

## Verification

```bash
# After Phase 1: gate is green
uv run pytest --cov=QDMpy --cov-report=term-missing
# Coverage should be ~55-60% with plotting excluded

# After Phase 3: target met
uv run pytest --cov=QDMpy --cov-report=term-missing
# Target: >80% total coverage (excluding plotting.py)

# Verify no unexplained skips
uv run pytest -v 2>&1 | grep SKIP  # only gpu/integration markers
```

## Rejection Alternatives

**Alternative: Ship full test dataset in the repo.** Rejected — ODMR datasets
are typically hundreds of MB. Even a minimal real dataset would bloat the repo.
Synthetic data is deterministic and fast.

**Alternative: Use a remote test data fixture (e.g. download from S3).** Adds
external dependency and network requirement to tests. Acceptable for integration
tests but not for the coverage gate.

**Alternative: Lower the coverage target permanently.** Rejected — 80% is a
reasonable target for scientific code. The problem is not the target but the
gap between target and reality.

**Alternative: Include plotting in the coverage target.** Rejected — plotting.py
is fundamentally broken (QEP-012) and including it would require fixing it first,
blocking progress on core coverage.

## Implementation Record (2026-02-19)

All phases were completed in a single session. Key decisions that differed from
the original plan:

### What was done

- **`tests/conftest.py`** — created with shared fixtures: `rng`, `raw_odmr_array`,
  `synthetic_odmr_data`, `synthetic_odmr`, `make_xr_data()`, `small_xr_data`,
  `synthetic_fit_manager`, `synthetic_fit_result`, `light_image`, `laser_image`

- **`tests/test_cli.py`** — new file covering all CLI entry points (parsers,
  subcommands, handlers, `main()`); CLI went from 0% → 100% coverage

- **`tests/test_init.py`** — new file covering `make_configfile()`, `reset_config()`,
  `reset_settings()`, `test_data_location()`, `is_pygpufit_available()`

- **`tests/odmr/test_io.py`** — added `TestMatlabLoaderEndToEnd` with 8 end-to-end
  tests using `scipy.io.savemat`-generated synthetic `.mat` files; io went from
  43% → 97%

- **`tests/test_fit.py`** — added `TestFitManagerUncoveredPaths` (8 tests for repr,
  unfitted-state errors, constraint expansion, GPU-unavailable paths)

- **`tests/test_result.py`** — added `TestComputeB111Branches` and
  `TestSaveResultsWithCaches` for previously uncovered delta_resonance shapes
  and save-with-cache behaviour

- **`tests/test_properties.py`** — new file with Hypothesis property-based tests
  for ESRSINGLE, ESR14N, ESR15N model functions and FitResult calculations

- **`# pragma: no cover`** added to all `@njit` decorated functions in `guess.py`
  (numba JIT-compiled to machine code; invisible to coverage.py), the
  `if __name__ == "__main__":` block in `measurement.py`, and
  `preview_fluorescence_correction()` in `odmr/processors.py`

- **Fixed `test_reference_images_loading`** — LED.csv uses tab-delimited format;
  changed `delimiter=","` → default whitespace delimiter

### Key implementation notes

- `model.func(freq, params)` returns `(N, len_freq)`; single-spectrum calls need
  `params.reshape(1, -1)` and `result.squeeze(axis=0)`
- MATLAB imgStack shape in `.mat` files is `(n_freqs, n_pixels)` before `.T`
- When `numFreqs != len(freqList)`, each imgStack has `n_per_range` rows (not
  the full `n_freqs_total`); the loader splits only the frequency list
- For ESR14N hypothesis tests, `width` must be < `AHYP/4 ≈ 5e-4 GHz` for
  the three hyperfine dips to remain resolved by `find_peaks`
- `FitResult._b_field_cache` and `_b111_cache` can be injected directly as
  Pydantic `PrivateAttr` fields; bypasses the reshape pipeline in tests

### Final metrics

```
390 passed, 33 skipped (intentional: integration/gpu/plotting), 0 failed
Total coverage: 93.61%  (gate: 80%)
```
