"""Integration tests: TorchBackend recovers ground truth like the gpufit kernels.

Mirror of ``test_gpufit_consistency.py`` for the torch backend (QEP-069):
noiseless synthetic spectra from known parameters are fitted and must satisfy
the same contract (all converged, chi2 < 1e-6, params within rtol=1e-2 /
atol=1e-5 columnwise-appropriate tolerances).

Two variants per model:
- from the true parameters (pure kernel-consistency, same as the gpufit test);
- from perturbed initial guesses (stronger — exercises the actual optimizer,
  which we own here, unlike gpufit's).

Devices: always 'cpu' (this is the CI path and shares 100% of the numeric
code with GPU devices); 'cuda'/'mps' are added when locally available so the
same file validates real GPUs on developer machines.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qdmpy.fitting.backends import FitBackendOptions

_HAS_TORCH = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch (gpu extra)")

N = 64
N_FREQ = 50
FREQ = np.linspace(2.82, 2.92, N_FREQ, dtype=np.float32)

_OPTIONS = FitBackendOptions(estimator="LSE", max_number_iterations=1000, tolerance=1e-10)

# (n_pixel, 2*n_params) bounds matching _GPUFIT_CONSISTENCY_SETTINGS in the
# gpufit twin test: center 2.82-2.92, width 1e-4..0.01, contrast 0..1,
# offset -0.1..0.1 — all LOWER_UPPER.
_BOUNDS_BY_TYPE = {
    "center": (2.82, 2.92),
    "width": (0.0001, 0.01),
    "contrast": (0.0, 1.0),
    "offset": (-0.1, 0.1),
}


def _devices() -> list[str]:
    devices = ["cpu"]
    import torch

    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _make_params(model_name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_contrast = {"ESR14N": 3, "ESR15N": 2, "ESRSINGLE": 1}[model_name]
    cols = [
        rng.uniform(2.85, 2.89, N),  # center (GHz)
        rng.uniform(0.002, 0.005, N),  # width (GHz)
        *[rng.uniform(0.05, 0.15, N) for _ in range(n_contrast)],
        rng.uniform(-0.01, 0.01, N),  # offset
    ]
    return np.column_stack(cols).astype(np.float32)


def _constraints_for(model) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for pname in model.parameter_names:
        lo, hi = _BOUNDS_BY_TYPE[model.parameter_types[pname]]
        pairs.extend((lo, hi))
    constraints = np.tile(np.array(pairs, dtype=np.float32), (N, 1))
    constraint_types = np.full(model.n_parameters, 3, dtype=np.int32)  # LOWER_UPPER
    return constraints, constraint_types


def _run_consistency(model_name: str, device: str, *, perturb: bool) -> None:
    from qdmpy.fitting.models import ModelRegistry
    from qdmpy.fitting.torch_backend import TorchBackend

    model = ModelRegistry.get(model_name)
    true_params = _make_params(model_name, seed=hash(model_name) % 2**31)
    spectra = model.func(FREQ, true_params).astype(np.float32)

    init = true_params.copy()
    if perturb:
        init[:, 0] += 0.001  # center off by ~1/3 linewidth
        init[:, 1] *= 1.3
        init[:, 2:-1] *= 0.7
        init[:, -1] += 0.005

    constraints, constraint_types = _constraints_for(model)
    out = TorchBackend(device=device).fit(
        spectra, FREQ, init, constraints, constraint_types, model, _OPTIONS
    )

    assert np.all(out.states == 0), (
        f"{model_name}/{device}: some fits did not converge — "
        f"states: {np.unique(out.states, return_counts=True)}"
    )
    assert np.all(out.chi2 < 1e-6), (
        f"{model_name}/{device}: nonzero chi2 — max {out.chi2.max():.2e}"
    )
    recovered = out.parameters
    assert_allclose(recovered[:, 0], true_params[:, 0], atol=5e-6)  # center (GHz)
    assert_allclose(recovered[:, 1], true_params[:, 1], rtol=2e-2)  # width
    assert_allclose(recovered[:, 2:-1], true_params[:, 2:-1], rtol=2e-2)  # contrasts
    assert_allclose(recovered[:, -1], true_params[:, -1], atol=1e-4)  # offset (~0)


@pytest.mark.parametrize("model_name", ["ESR14N", "ESR15N", "ESRSINGLE"])
def test_recovers_truth_from_true_start(model_name: str) -> None:
    for device in _devices():
        _run_consistency(model_name, device, perturb=False)


@pytest.mark.parametrize("model_name", ["ESR14N", "ESR15N", "ESRSINGLE"])
def test_recovers_truth_from_perturbed_start(model_name: str) -> None:
    for device in _devices():
        _run_consistency(model_name, device, perturb=True)
