# Integration tests

Three modules live here. "Integration" in this directory means **needs a real
GPU or a real dataset** — that is the distinction that decides whether a test
can run at all, not whether it spans several components. Tests that exercise
the full pipeline against synthetic data are plain unit tests and live in
`tests/`.

| Module | What it checks | Gate |
|---|---|---|
| `test_gpufit_consistency.py` | The Python ESR models and the pyGpufit CUDA kernels agree: noiseless spectra generated from known parameters, fitted from the truth, must give chi2 ~ 0. Model IDs are resolved from `pygpufit.gpufit.ModelID` so a version mismatch is caught rather than assumed. | CUDA |
| `test_torch_consistency.py` | The same contract for `TorchBackend` (QEP-069). Tolerances were tightened by 2-4 orders of magnitude when the models gained analytic Jacobians (QEP-073) and sit ~10-50x above what the backend achieves, so a regression to finite differences fails. | torch |
| `test_folded_real_data_regression.py` | Folded fitting against cropped real FOVs: folded and normal paths must agree on B111. | CUDA + real data |

## The two gates

**CUDA.** `_HAS_GPUFIT = _gf.cuda_available()` — an actual runtime probe, not an
import check. pyGpufit imports fine anywhere the wheel is installed, since it is
a thin wrapper around a compiled CUDA library; it only fails at fit time without
a working driver. Note that since v0.1.0 pyGpufit is an optional extra, so it is
absent unless you `uv sync --extra gpufit`.

**Real data.** `tests/data/` is **gitignored** — the smallest FOV is 61 MB and
pre-commit caps files at 1500 kB, so it cannot be committed. These tests
therefore **skip on CI by design**, and that is not a failure to fix. Modules
needing it carry `@pytest.mark.requires_real_data`.

Because of those gates, a green run on CI says nothing about the GPU backends.
`MatlabLoader.load()` is covered on CI by the synthetic `.mat` files in
`tests/odmr/test_io.py`, which build real MATLAB v5 files in `tmp_path`.

## Running them

```sh
uv run pytest                              # integration modules skip if gated
uv run pytest -m integration               # just this directory
uv run pytest -m "not requires_real_data"  # everything that needs no local dataset
uv run pytest -m "not integration"         # the CI-equivalent selection
```

Markers are registered in `pyproject.toml` and every one of them selects a
non-zero number of tests; `tests/test_markers.py` enforces that, and
`--strict-markers` rejects a typo at collection time.

## Reference NPZ files

`reference_data/FOV18x_reference_bin{2,6}.npz` are **not read by any test**. They
are leftovers from the old-vs-new comparison harness that was removed once
QEP-031 identified its root cause (AHYP is hardcoded in GHz in the gpufit
kernels). They are kept because QEP-073 cites them for real fitted widths.
Rebuilding a regression test on them would be new work, not a restoration.
