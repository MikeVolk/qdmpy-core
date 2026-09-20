# QEP-TEST-001 — Test Foundation Fixes

**Status:** Implemented (2026-09-20)
**Created:** 2026-02-22
**Revised:** 2026-09-20 -- drift update; H-11 premise superseded, H-12 premise
confirmed on CI but masked locally; Proposed Changes Phase 2 obsolete
**Severity:** HIGH (H-10, H-11, H-12)
**Module:** `tests/`

---

## Status (2026-09-20)

Implemented, but two of the three problems had drifted since 2026-02-22 and the
work done differs from the Proposed Changes below. The original text is kept
intact -- the drift record is the point -- with the corrections here.

### H-10 -- unchanged, fixed as written

`tests/test_fit.py` still carried `pol_0`/`frange_0`. Worth recording *why* it
never failed: `FitManager` indexes positionally, so the wrong labels were
invisible. Data built by that helper could not reach `odmr/analysis.py`,
`odmr/folding.py` or `fitting/result.py:402` at all -- `b111_from_dip_positions`
raises `DataValidationError` on it. The fixture was structurally incapable of
catching a label bug, and one of the two duplicated B111 paths was therefore
untestable.

Fixed by `tests/helpers.py`, which sources labels from `qdmpy.constants` and
replaces **three** duplicate builders (the QEP knew of two; a third lived at
`tests/test_load.py:24`). Because helpers now import the constants, a rename
would propagate silently, so `test_constants.py` locks the literal values.

### H-11 -- premise superseded

Commit `f48e0c5` fixed this before the QEP was actioned. `validation_tests.utils`,
`new_qdmpy_modules`, `test_data_folder` and `QDMPY_TEST_DATA` do not exist, and
no test skips for missing data. **Proposed Changes Phase 2 below is obsolete and
was not implemented** -- do not resurrect it.

The real problem in its place: all 11 markers were registered and **zero were
applied**, so every `-m` command in `tests/integration/README.md` selected
nothing, and that README documented six fixtures, five test files and a
generator script that do not exist. Markers are now applied module-wide;
`validation` and `performance` were dropped as unapplicable and
`requires_reference_data` renamed `requires_real_data`; `--strict-markers` and
`tests/test_markers.py` prevent recurrence.

### H-12 -- true on CI, masked locally

The premise looks fixed locally and is not. `tests/data/` is **gitignored**
(`.gitignore:112`, zero tracked files; the smallest FOV is 61 MB against a
1500 kB pre-commit cap), so `MatlabLoader.load()` had **no CI coverage at all**
-- a local "1051 passed" was flattered by machine-local data. Record this: it
will otherwise be re-discovered and wrongly re-closed.

Also found: `test_keys_missing_exception` was a tautology. It re-implemented
`io.py`'s try/except inside its own body and asserted its own raise, never
calling `MatlabLoader`; coverage showed `io.py:134-136` and `154-156` uncovered
for its entire life. Replaced with synthetic-`.mat` tests that drive `load()`
end to end, verified by mutation rather than coverage alone.

`testing.py`'s 13% figure is long stale (it was 93%), but the substance held:
zero of its seven public symbols had a direct test, and `make_synthetic_fit_result`
had none at all despite two tutorial notebooks depending on it.

### Outcome

| | Before | After |
|---|---|---|
| Tests | 1061 passed, 26 skipped | 1106 passed, 26 skipped |
| `odmr/io.py` | 88% | 97% |
| `qdmpy/testing.py` | 93% | 100% |
| Markers applied | 0 of 11 | 9 of 9 |

The 26 skips are correct optional-dependency gating (CUDA and torch) and must
stay -- the Test Plan's "0 skipped" criterion below was wrong.

Beyond the QEP's scope but enabled by it: `tests/test_b111_parity.py` pins
agreement between the two duplicated B111 implementations as a characterisation
test for QEP-FIT-004.

## GUI Integration Requirements

**Impact: none.** This QEP touches `tests/`, `pyproject.toml`
`[tool.pytest.ini_options]`, `CHANGELOG.md` and two READMEs. No file under
`src/qdmpy/` changed, so there is no core API, data contract, settings key,
map/result field or persisted format for `qdmpy-gui` to track, and no migration.

1. **Core API touchpoints:** none. `qdmpy.testing` is imported by the GUI and is
   unchanged -- this QEP adds tests *for* it without altering it.
2. **State/settings migration:** none; no defaults, keys or persisted data changed.
3. **Progress/warning/error behavior:** unchanged; no new user-facing conditions.
4. **Acceptance check:** `uv run pytest tests/ -q` in `qdmpy-gui` passes
   unchanged against this branch of `qdmpy-core`.
5. **Rationale for no impact:** verified by `git diff --stat develop..HEAD`
   showing no `src/qdmpy/` path.

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

### Phase 2: Integration test activation (H-11) -- OBSOLETE, NOT IMPLEMENTED

> Superseded by commit `f48e0c5`. None of the symbols below exist any more.
> Kept as a record of what was believed in 2026-02. See Status (2026-09-20).

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

- [x] Phase 1: All fitting tests pass with canonical polarity labels
- [~] Phase 2: ~~`pytest` shows `0 skipped`~~ -- **wrong criterion.** The 26
      skips are correct optional-dependency gating (CUDA, torch) and must stay.
      Replaced by: every registered marker selects a non-zero number of tests.
- [x] Phase 2: `pytest -m integration` selects the integration modules (16 items)
- [x] Phase 3: `MatlabLoader.load()` covered end to end, and on CI, via
      synthetic `.mat` files; verified by mutation, not coverage alone
- [x] Phase 3: `testing.py` at 100% with a direct test per public symbol
