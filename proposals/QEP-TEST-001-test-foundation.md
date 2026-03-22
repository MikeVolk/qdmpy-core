# QEP-TEST-001 — Test Foundation Fixes

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-10, H-11, H-12)
**Module:** `tests/`

---

## Motivation

Three issues undermine the reliability of the test suite:

1. **H-10: `test_fit.py` uses pre-QEP-025 polarity labels.** The private
   `_make_xr_data()` helper in `tests/test_fit.py:75` sets polarity coords to
   `pol_0`/`pol_1` instead of the canonical `neg`/`pos` (established in
   QEP-025). Any test that exercises polarity-labelled operations
   (`.sel(polarity='neg')`) on data from this helper will silently produce
   wrong results or confusing `KeyError` failures.

2. **H-11: All integration tests permanently skip.** Every integration test
   depends on:
   - `tests/data/FOV18x/` existing on the filesystem, or
   - `validation_tests.utils` being importable (always fails).

   Neither condition is documented. Tests show as "51 skipped" — the suite
   appears green while the integration layer is actually untested. The
   `test_data_folder` session fixture calls `pytest.skip()` inline, which is
   a pytest anti-pattern for session-scoped fixtures.

3. **H-12: `MatlabLoader.load()` happy path is untested.** The unit tests in
   `tests/odmr/test_io.py` test only `__init__`, error branches, and the
   static `_process_mat_file_*` helpers. The actual `.load()` path that parses
   `.mat` files, splits frequencies, applies `.T`, and builds the xarray is
   only exercised by the permanently-skipping integration tests.

   Similarly, `testing.py` (public testing helpers) has only 13% coverage.

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Proposed Changes

### Phase 1: Fix polarity labels (H-10)

Delete `_make_xr_data()` from `tests/test_fit.py`. Use the shared
`make_xr_data()` from `tests/conftest.py` which already uses canonical
`neg`/`pos` labels:

```python
# tests/test_fit.py — before
from tests.test_fit import _make_xr_data  # private duplicate

# tests/test_fit.py — after
from tests.conftest import make_xr_data   # shared, correct labels
```

Also fix `freq_range` labels from `frange_0`/`frange_1` to `low`/`high`.

### Phase 2: Integration test activation (H-11)

1. Add a `pytest.mark.integration` marker:

```python
# conftest.py or pyproject.toml
markers = ["integration: requires real QDM data files"]
```

2. Rewrite `test_data_folder` fixture to use the marker:

```python
@pytest.fixture(scope='session')
def test_data_folder():
    path = Path(os.environ.get('QDMPY_TEST_DATA', 'tests/data/FOV18x'))
    if not path.exists():
        path = Path.home() / 'Documents' / 'FOV18x'
    if not path.exists():
        pytest.skip('No test data: set QDMPY_TEST_DATA or place data in tests/data/FOV18x')
    return path
```

3. Mark all integration tests:

```python
@pytest.mark.integration
class TestFullPipeline:
    ...
```

4. In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["integration: requires real QDM data files"]
```

5. Remove dead `validation_tests.utils` imports and `new_qdmpy_modules`
   fixture from `tests/integration/conftest.py`.

6. Document in README/CONTRIBUTING:

```
# Run unit tests only (default)
uv run pytest

# Run with integration tests (requires data)
QDMPY_TEST_DATA=/path/to/FOV18x uv run pytest -m 'integration or not integration'
```

### Phase 3: MatlabLoader and testing.py coverage (H-12)

1. **Synthetic `.mat` test for `MatlabLoader.load()`:**

```python
# tests/odmr/test_io.py
def test_matlabloader_load_synthetic(tmp_path):
    """Test MatlabLoader with a synthesized .mat file."""
    from scipy.io import savemat

    n_freq = 20
    n_pixels = 100
    freq_list = np.linspace(2.8e9, 3.0e9, 2 * n_freq)

    for i, name in enumerate(['run_00000.mat', 'run_00001.mat']):
        savemat(tmp_path / name, {
            'imgStack1': np.random.default_rng(i).random((n_freq, n_pixels)),
            'imgStack2': np.random.default_rng(i + 10).random((n_freq, n_pixels)),
            'imgNumRows': 10,
            'imgNumCols': 10,
            'freqList': freq_list,
            'numFreqs': n_freq,
        })

    loader = MatlabLoader(str(tmp_path))
    result = loader.load()

    assert result.dims == ('polarity', 'freq_range', 'y', 'x', 'freq_idx')
    assert result.sizes['polarity'] == 2
    assert result.sizes['freq_range'] == 2
    assert result.sizes['y'] == 10
    assert result.sizes['x'] == 10
    assert result.sizes['freq_idx'] == n_freq
    assert 'freq_ghz' in result.coords
```

2. **Tests for `testing.py` public helpers:**

```python
# tests/test_testing_helpers.py
from QDMpy.testing import make_synthetic_fit_result, make_synthetic_odmr_data

def test_make_synthetic_fit_result():
    result = make_synthetic_fit_result()
    assert 'center' in result.parameters
    assert result.b111 is not None

def test_make_synthetic_odmr_data():
    data = make_synthetic_odmr_data()
    assert data.data.dims == ('polarity', 'freq_range', 'y', 'x', 'freq_idx')
```

## Migration

- Phase 1: May require updating test assertions that reference `pol_0`/`frange_0`.
- Phase 2: Existing integration test runs unchanged (`pytest` skips by default).
- Phase 3: New test files, no migration needed.

## Test Plan

- [ ] Phase 1: All fitting tests pass with canonical polarity labels
- [ ] Phase 2: `pytest` shows `0 skipped` for unit tests (integration excluded by default)
- [ ] Phase 2: `pytest -m integration` runs integration tests when data present
- [ ] Phase 3: `MatlabLoader.load()` unit test covers happy path
- [ ] Phase 3: `testing.py` helpers have >80% coverage
