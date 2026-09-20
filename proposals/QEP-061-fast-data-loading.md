# QEP-061 -- Fast Data Loading

| Field   | Value                     |
|---------|---------------------------|
| Status  | Rejected                  |
| Created | 2026-03-10                |
| Rejected | 2026-03-10               |
| Scope   | `qdmpy.odmr.io`, `qdmpy.io.images` |

## Motivation

Loading a QDM dataset from MATLAB files is the single slowest step before
fitting.  For a typical 2k x 2k scan with 2 polarities, `MatlabLoader.load()`
spends 1--8 seconds reading 2--4 `.mat` files sequentially, followed by
~100--500 ms parsing CSV reference images via `np.loadtxt()`.  Processing
(binning, normalization, fluorescence correction) adds < 1 s total.

This QEP targets two independent, low-risk optimizations to reduce cold-load
time without changing any public API.

## Design

### Phase 1: Parallel `.mat` file loading

**File:** `src/qdmpy/odmr/io.py`

**Problem:** `MatlabLoader.load()` reads each `run_*.mat` file sequentially in
a `for file in files` loop (line 90).  The files are independent -- each
produces one `(n_frange, n_pixels, n_freqs)` array plus shared metadata.

**Solution:** Use `concurrent.futures.ThreadPoolExecutor` to load files in
parallel.  Both `mat73.loadmat()` (h5py-backed) and `scipy.io.loadmat()`
release the GIL during I/O, so threads provide real concurrency.

```python
def _load_single_mat(self, file: str) -> tuple[NDArray, dict[str, Any]]:
    """Load one .mat file, returning (stacked_data, raw_mat_dict)."""
    full_path = os.path.join(self.data_folder, file)
    try:
        mat_data = mat73.loadmat(full_path)
    except (TypeError, OSError):
        mat_data = loadmat(full_path)
    stacked = self._process_mat_file(mat_data)
    return stacked, mat_data
```

In `load()`, replace the sequential loop with:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=len(files)) as pool:
    results = list(pool.map(self._load_single_mat, files))

per_file_data = [r[0] for r in results]
# Extract metadata from the last file (they all share dims/freqs)
mat_data = results[-1][1]
```

Metadata extraction (rows, cols, frequencies) uses the last file's dict, same
as the current code does implicitly.

**Expected gain:** ~50% wall-time reduction for 2-file datasets, ~65% for
4-file datasets (I/O overlap).

### Phase 2: Fast CSV image parsing

**File:** `src/qdmpy/io/images.py`

**Problem:** `get_image()` (line 93) uses `np.loadtxt(file_path, delimiter=",")`
which is pure-Python text parsing -- slow for 2k x 2k grids (~100--500 ms).
A whitespace-fallback path adds a second `np.loadtxt()` call on failure.

**Solution:** Replace with `numpy.genfromtxt()` using `dtype=np.float64` and
explicit delimiter, or better, use `pandas.read_csv()` with C engine then
extract `.values`.  However, to avoid adding a pandas dependency just for
image loading, use `numpy.fromfile()` for binary or the newer
`numpy.loadtxt()` improvements in NumPy >= 2.0 (C-accelerated parser).

Preferred approach -- keep zero new dependencies:

```python
# NumPy >= 1.23 has a fast C parser for loadtxt
img = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
```

NumPy >= 1.23 already uses a C-based parser under the hood for `loadtxt`.
The actual fix is to avoid the double-parse on whitespace fallback by probing
the delimiter first:

```python
def _detect_delimiter(file_path: str) -> str | None:
    """Read first line to detect comma vs whitespace delimiter."""
    with open(file_path) as f:
        first_line = f.readline()
    return ',' if ',' in first_line else None


def _load_csv_image(file_path: str) -> NDArray:
    """Load a CSV/whitespace image file with fast delimiter detection."""
    delimiter = _detect_delimiter(file_path)
    return np.loadtxt(file_path, delimiter=delimiter, dtype=np.float64)
```

This eliminates the try/except double-parse and lets NumPy's C parser handle
both formats in a single pass.

**Expected gain:** ~2--3x faster CSV parsing; eliminates redundant parse on
whitespace-delimited files.

### Phase 3: HDF5 lazy reading for `.mat` files (evaluated)

**File:** `src/qdmpy/odmr/io.py`

**Problem:** `mat73.loadmat()` eagerly reads the entire HDF5 file into memory,
including all image stacks. For v7.3 `.mat` files, this means decompressing
and copying all data before processing.

**Proposed solution:** Use `h5py` directly for v7.3 files, reading datasets
with direct key access and preserving the existing `scipy.io.loadmat()`
fallback for non-v7.3 files.

**Expected gain (proposal):** ~20--40% faster per-file read for v7.3 files.

## Implementation Order

Phase 1 and 2 are independent and can be done in parallel.

```
Phase 1: Parallel .mat loading   ──┐
Phase 2: Fast CSV image parsing   ──┤
Tests & benchmarks ─────────────────┘
```

## Testing

- **Unit tests:** mock `mat73.loadmat` / `h5py.File` to verify parallel
  dispatch and correct metadata extraction order.
- **Regression:** load `reference_data/FOV18x_reference_bin2.npz` fixture;
  verify identical processed data arrays (bitwise).
- **Benchmark script:** `scripts/bench_load.py` that times `from_folder()`
  on a real dataset before/after, reporting wall-time and peak RSS.

## Evaluation Results

Benchmarks and prototype checks were run against `~/Documents/FOV1`
(2 x v7.3 files, ~1.54 GB each) and repository test data.

| Candidate | Result | Evidence |
|----------|--------|----------|
| Phase 1: threaded `.mat` loading | Regressed | Current loader: 26.0s; threaded prototype: 37.2s (0.70x) on `~/Documents/FOV1` |
| Phase 2: delimiter detection for CSV | Small positive | `LED.csv`: 0.138s -> 0.120s; `laser.csv`: 0.143s -> 0.132s |
| Phase 3: direct `h5py` read (as proposed) | Incorrect | Fails reshape on `~/Documents/FOV1` due to stack orientation mismatch |
| Phase 3: `h5py` with transpose compatibility fix | Correct but slower | Baseline loader: 27.1s; `h5py` fixed prototype: 35.5s; outputs identical |

Additional observations:
- `h5py` and `mat73` both read v7.3 test fixtures successfully.
- Direct `h5py` arrays for `imgStack*` are shaped `(n_pixels, n_freqs)`, while
  `mat73` returns `(n_freqs, n_pixels)` on tested files.
- `scipy.io.loadmat()` correctly handles non-v7.3 test fixtures and rejects
  v7.3 fixtures as expected.

## Rejection Rationale

QEP-061 is rejected because measured outcomes do not support the central goal
of materially reducing end-to-end load time:

- The primary optimization (Phase 1) causes a substantial slowdown on a real
  production-scale dataset.
- The highest-risk optimization (Phase 3) is invalid as written and remains
  slower than baseline after correctness fixes.
- The only positive change (Phase 2) is a low-risk micro-optimization with a
  small absolute impact that does not justify the full QEP by itself.

Future work should start from a measurement-first proposal (step-level
profiling and cache strategy), then target the dominant cost center.

## Risks

| Risk | Mitigation |
|------|------------|
| Thread-safety of `mat73` / `scipy.io` | Both release GIL; no shared mutable state between file reads |
| Delimiter detection false positive | Only check first non-comment line; CSV images from QDM are always uniform format |
| Behavioral change in load order | Metadata extracted from last file (alphabetically) -- same as current implicit behavior; add assertion that all files share dims |

## GUI Integration Requirements

No GUI changes required.  `Measurement.from_folder()` is the sole entry point
and its signature/return type are unchanged.  The GUI's `LoadWorker` will
observe faster completion times transparently.

## Out of Scope

- Direct `h5py`-based v7.3 `.mat` loading for production path
- Zarr/chunked storage format (future QEP for out-of-core workflows)
- GPU-resident intermediate data between fold and fit
- Adaptive float16 precision
