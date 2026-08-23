"""Integration tests: verify Python ESR models match the pygpufit GPU kernels.

Strategy: generate noiseless synthetic spectra from known parameters using
the Python model, then fit them with gpufit starting from the true parameters.
If the Python and GPU implementations agree, chi2 will be ≈ 0 and recovered
parameters will match ground truth within float32 precision.

Model IDs are resolved dynamically from ``pygpufit.gpufit.ModelID`` so any
mismatch between our hardcoded IDs and the installed package version is caught
immediately — without relying on hardcoded integers in the test itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qdmpy.settings import FitSettings, ModelConstraintsSettings, ModelSettings, QDMpySettings

try:
    import pygpufit.gpufit as gf

    _HAS_GPUFIT = True
except (ImportError, OSError):
    gf = None
    _HAS_GPUFIT = False

pytestmark = pytest.mark.skipif(not _HAS_GPUFIT, reason="Requires pygpufit installation")

N = 64
N_FREQ = 50
FREQ = np.linspace(2.82, 2.92, N_FREQ, dtype=np.float32)

# Maps our model registry name → the corresponding pygpufit ModelID attribute name.
_MODEL_ID_ATTR: dict[str, str] = {
    "ESR14N": "ESR14N",
    "ESR15N": "ESR15N",
    "ESRSINGLE": "ESRSINGLE",
}

_GPUFIT_CONSISTENCY_SETTINGS = QDMpySettings(
    fit=FitSettings(estimator="LSE", max_number_iterations=1000, tolerance=1e-10),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.82,
            center_max=2.92,
            center_type="LOWER_UPPER",
            width_min=0.0001,
            width_max=0.01,
            width_type="LOWER_UPPER",
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type="LOWER_UPPER",
            offset_min=-0.1,
            offset_max=0.1,
            offset_type="LOWER_UPPER",
        )
    ),
)


def _expected_model_id(model_name: str) -> int:
    """Return the model ID from the installed pygpufit package for *model_name*."""
    assert gf is not None
    attr = _MODEL_ID_ATTR[model_name]
    return int(getattr(gf.ModelID, attr))


@pytest.mark.parametrize("model_name", list(_MODEL_ID_ATTR))
def test_model_id_matches_package(model_name: str) -> None:
    """Our hardcoded model_id must equal the value in pygpufit.gpufit.ModelID."""
    from qdmpy.fitting.models import ModelRegistry

    model = ModelRegistry.get(model_name)
    expected = _expected_model_id(model_name)
    assert model.model_id == expected, (
        f"{model_name}: model.model_id={model.model_id} but "
        f"pygpufit.gpufit.ModelID.{model_name}={expected}. "
        f"Update models.py or upgrade/downgrade pygpufit."
    )


def _run_consistency(model_name: str, true_params: np.ndarray) -> None:
    """Fit noiseless synthetic spectra and assert GPU/Python implementations agree."""
    from qdmpy.fitting.manager import FitManager
    from qdmpy.fitting.models import ModelRegistry

    model = ModelRegistry.get(model_name)
    spectra = model.func(FREQ, true_params).astype(np.float32)  # (N, n_freq)

    fm = FitManager(
        model_name=model_name, settings=_GPUFIT_CONSISTENCY_SETTINGS, gpu_available=True
    )
    data = spectra[np.newaxis]  # (1, N, n_freq)
    init = true_params[np.newaxis]  # (1, N, n_params)

    results = fm.fit_frange(data, FREQ, init, irange=0, n_frange=1)
    recovered = results[0].reshape(-1, model.n_parameters)
    states = results[1].flatten()
    chi2 = results[2].flatten()

    assert np.all(states == 0), (
        f"{model_name}: some fits did not converge — "
        f"states: {np.unique(states, return_counts=True)}"
    )
    assert np.all(chi2 < 1e-6), (
        f"{model_name}: nonzero chi2 suggests model mismatch — max chi2={chi2.max():.2e}"
    )
    assert_allclose(
        recovered,
        true_params,
        rtol=1e-2,
        atol=1e-5,
        err_msg=f"{model_name}: recovered params differ from ground truth",
    )


def test_esr14n_matches_gpufit() -> None:
    """Python ESR14N must match the GPU kernel (model ID resolved from package)."""
    rng = np.random.default_rng(0)
    params = np.empty((N, 6), dtype=np.float32)
    params[:, 0] = rng.uniform(2.85, 2.89, N)  # center (GHz)
    params[:, 1] = rng.uniform(0.002, 0.005, N)  # width (GHz)
    params[:, 2] = rng.uniform(0.05, 0.15, N)  # contrast_0
    params[:, 3] = rng.uniform(0.05, 0.20, N)  # contrast_1
    params[:, 4] = rng.uniform(0.05, 0.15, N)  # contrast_2
    params[:, 5] = rng.uniform(-0.01, 0.01, N)  # offset
    _run_consistency("ESR14N", params)


def test_esr15n_matches_gpufit() -> None:
    """Python ESR15N must match the GPU kernel (model ID resolved from package)."""
    rng = np.random.default_rng(1)
    params = np.empty((N, 5), dtype=np.float32)
    params[:, 0] = rng.uniform(2.85, 2.89, N)
    params[:, 1] = rng.uniform(0.002, 0.005, N)
    params[:, 2] = rng.uniform(0.05, 0.20, N)
    params[:, 3] = rng.uniform(0.05, 0.20, N)
    params[:, 4] = rng.uniform(-0.01, 0.01, N)
    _run_consistency("ESR15N", params)


def test_esrsingle_matches_gpufit() -> None:
    """Python ESRSINGLE must match the GPU kernel (model ID resolved from package)."""
    rng = np.random.default_rng(2)
    params = np.empty((N, 4), dtype=np.float32)
    params[:, 0] = rng.uniform(2.85, 2.89, N)
    params[:, 1] = rng.uniform(0.002, 0.005, N)
    params[:, 2] = rng.uniform(0.05, 0.30, N)
    params[:, 3] = rng.uniform(-0.01, 0.01, N)
    _run_consistency("ESRSINGLE", params)
