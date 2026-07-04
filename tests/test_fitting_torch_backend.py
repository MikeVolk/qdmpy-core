"""Unit tests for the TorchBackend batched-LM engine (QEP-069).

Tests that must run WITHOUT torch installed live in TestWithoutTorch (they
patch availability); everything else is skipped cleanly when torch is
missing so the suite still passes on a torch-less install.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from qdmpy.exceptions import DependencyError
from qdmpy.fitting.backends import FitBackendOptions, bounds_from_constraints
from qdmpy.fitting.manager import FitManager
from qdmpy.fitting.models import Model, ModelRegistry
from qdmpy.fitting.torch_backend import TorchBackend, torch_gpu_device_available
from qdmpy.settings import FitSettings, ModelConstraintsSettings, ModelSettings, QDMpySettings
from qdmpy.testing import make_synthetic_odmr_data

_HAS_TORCH = importlib.util.find_spec("torch") is not None

MOCK_SETTINGS = QDMpySettings(
    fit=FitSettings(estimator="LSE", max_number_iterations=200, tolerance=1e-8),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.80,
            center_max=2.90,
            center_type="LOWER_UPPER",
            width_min=0.0005,
            width_max=0.01,
            width_type="LOWER_UPPER",
            contrast_min=0.001,
            contrast_max=1.0,
            contrast_type="LOWER_UPPER",
            offset_min=-0.5,
            offset_max=0.5,
            offset_type="FREE",
        )
    ),
)


def _synthetic_esrsingle(n: int, seed: int = 0, noise: float = 0.0):
    """Noiseless-or-noisy ESRSINGLE spectra with known params and perturbed x0."""
    rng = np.random.default_rng(seed)
    model = ModelRegistry.get("ESRSINGLE")
    x = np.linspace(2.83, 2.86, 50)
    true = np.column_stack(
        [
            rng.uniform(2.840, 2.852, n),
            rng.uniform(0.002, 0.004, n),
            rng.uniform(0.05, 0.15, n),
            rng.uniform(-0.01, 0.01, n),
        ]
    ).astype(np.float32)
    y = model.func(x, true).astype(np.float32)
    if noise:
        y = y + rng.normal(0, noise, y.shape).astype(np.float32)
    x0 = true * np.array([1.0, 1.3, 0.7, 1.0], dtype=np.float32)
    x0[:, 0] += 0.001
    return model, x, y, true, x0


def _free_constraints(n: int, n_params: int):
    constraints = np.zeros((n, 2 * n_params), dtype=np.float32)
    constraint_types = np.zeros(n_params, dtype=np.int32)  # FREE
    return constraints, constraint_types


class TestWithoutTorch:
    """Behavior when torch is not installed (mocked away)."""

    def test_is_available_false(self) -> None:
        with patch("qdmpy.fitting.torch_backend.importlib.util.find_spec", return_value=None):
            assert TorchBackend().is_available() is False

    def test_gpu_device_available_false(self) -> None:
        with patch("qdmpy.fitting.torch_backend.importlib.util.find_spec", return_value=None):
            assert torch_gpu_device_available() is False

    def test_fit_raises_naming_gpu_extra(self) -> None:
        model, x, y, _true, x0 = _synthetic_esrsingle(2)
        constraints, ctypes = _free_constraints(2, 4)
        with (
            patch.dict(sys.modules, {"torch": None}),
            pytest.raises(DependencyError, match="--extra gpu"),
        ):
            TorchBackend().fit(y, x, x0, constraints, ctypes, model, FitBackendOptions())


@pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch (gpu extra)")
class TestDeviceResolution:
    """Device auto-selection and explicit-device validation."""

    def test_auto_prefers_cuda(self) -> None:
        import torch

        backend = TorchBackend(device="auto")
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            assert backend._resolve_device(torch).type == "cuda"

    def test_auto_falls_back_to_mps_then_cpu(self) -> None:
        import torch

        backend = TorchBackend(device=None)
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            assert backend._resolve_device(torch).type == "mps"
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            assert backend._resolve_device(torch).type == "cpu"

    def test_explicit_unavailable_cuda_raises(self) -> None:
        import torch

        backend = TorchBackend(device="cuda")
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            pytest.raises(DependencyError, match="cuda"),
        ):
            backend._resolve_device(torch)


@pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch (gpu extra)")
class TestLevenbergMarquardt:
    """Numerical behavior of the batched LM engine (CPU device)."""

    def test_recovers_known_lorentzian(self) -> None:
        model, x, y, true, x0 = _synthetic_esrsingle(64)
        constraints, ctypes = _free_constraints(64, 4)
        out = TorchBackend(device="cpu").fit(
            y, x, x0, constraints, ctypes, model, FitBackendOptions(estimator="LSE")
        )
        assert np.all(out.states == 0)
        assert np.all(out.chi2 < 1e-6)
        fitted = out.parameters
        np.testing.assert_allclose(fitted[:, 0], true[:, 0], atol=5e-6)  # center (GHz)
        np.testing.assert_allclose(fitted[:, 1], true[:, 1], rtol=2e-2)  # width
        np.testing.assert_allclose(fitted[:, 2], true[:, 2], rtol=2e-2)  # contrast
        np.testing.assert_allclose(fitted[:, 3], true[:, 3], atol=1e-4)  # offset (~0)

    def test_constraint_clamping(self) -> None:
        """LOWER_UPPER center bounds keep the result inside the box; FREE ignored."""
        model, x, y, _true, x0 = _synthetic_esrsingle(16)
        n_params = 4
        constraints = np.tile(
            np.array([2.8455, 2.8470, 0.0, 0.01, 0.0, 1.0, -1.0, 1.0], dtype=np.float32),
            (16, 1),
        )
        # center LOWER_UPPER; width FREE (bounds present but must be ignored);
        # contrast LOWER_UPPER; offset FREE
        ctypes = np.array([3, 0, 3, 0], dtype=np.int32)
        out = TorchBackend(device="cpu").fit(
            y, x, x0, constraints, ctypes, model, FitBackendOptions(estimator="LSE")
        )
        centers = out.parameters[:, 0]
        assert np.all(centers >= 2.8455 - 1e-6)
        assert np.all(centers <= 2.8470 + 1e-6)

    def test_initial_params_outside_bounds_are_clipped(self) -> None:
        """gpufit-parity: out-of-bounds x0 must not error (clamped instead)."""
        model, x, y, _true, x0 = _synthetic_esrsingle(8)
        x0[:, 0] = 5.0  # far outside the center window
        constraints = np.tile(
            np.array([2.84, 2.86, 0.001, 0.01, 0.001, 1.0, -1.0, 1.0], dtype=np.float32),
            (8, 1),
        )
        ctypes = np.array([3, 3, 3, 3], dtype=np.int32)
        out = TorchBackend(device="cpu").fit(
            y, x, x0, constraints, ctypes, model, FitBackendOptions(estimator="LSE")
        )
        assert np.all(out.parameters[:, 0] <= 2.86 + 1e-6)

    def test_chunked_equals_unchunked(self) -> None:
        model, x, y, _true, x0 = _synthetic_esrsingle(23, noise=0.001)
        constraints, ctypes = _free_constraints(23, 4)
        opts = FitBackendOptions(estimator="LSE")
        out_small = TorchBackend(device="cpu", chunk_size=7).fit(
            y, x, x0, constraints, ctypes, model, opts
        )
        out_big = TorchBackend(device="cpu", chunk_size=10**6).fit(
            y, x, x0, constraints, ctypes, model, opts
        )
        np.testing.assert_array_equal(out_small.parameters, out_big.parameters)
        np.testing.assert_array_equal(out_small.states, out_big.states)

    def test_nan_data_gives_state_2(self) -> None:
        model, x, y, _true, x0 = _synthetic_esrsingle(4)
        y[1] = np.nan
        constraints, ctypes = _free_constraints(4, 4)
        out = TorchBackend(device="cpu").fit(
            y, x, x0, constraints, ctypes, model, FitBackendOptions(estimator="LSE")
        )
        assert out.states[1] == 2
        assert np.all(out.states[[0, 2, 3]] == 0)

    def test_output_contract(self) -> None:
        model, x, y, _true, x0 = _synthetic_esrsingle(6)
        constraints, ctypes = _free_constraints(6, 4)
        out = TorchBackend(device="cpu").fit(
            y, x, x0, constraints, ctypes, model, FitBackendOptions(estimator="LSE")
        )
        assert out.parameters.dtype == np.float32
        assert out.parameters.shape == (6, 4)
        assert out.states.dtype == np.int32
        assert out.chi2.dtype == np.float32
        assert out.iterations.dtype == np.int32
        assert np.all(out.iterations >= 1)
        assert out.execution_time > 0

    def test_mle_falls_back_to_lse(self) -> None:
        model, x, y, true, x0 = _synthetic_esrsingle(4)
        constraints, ctypes = _free_constraints(4, 4)
        out = TorchBackend(device="cpu").fit(
            y, x, x0, constraints, ctypes, model, FitBackendOptions(estimator="MLE")
        )
        assert np.all(out.states == 0)
        np.testing.assert_allclose(out.parameters, true, rtol=1e-2, atol=1e-5)


@pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch (gpu extra)")
class TestFrameworkNeutralContract:
    """Custom-model support and the framework-neutrality guard."""

    def test_custom_pure_python_model_end_to_end(self) -> None:
        name = "_QEP069_TORCH_CUSTOM"
        if name not in ModelRegistry.all():

            @ModelRegistry.register
            class _Custom(Model):
                name: ClassVar[str] = "_QEP069_TORCH_CUSTOM"

                def __init__(self) -> None:
                    super().__init__(
                        "_QEP069_TORCH_CUSTOM", 1, ["center", "width", "contrast", "offset"]
                    )
                    self.model_id = -1

                @property
                def parameter_types(self) -> dict[str, str]:
                    return {
                        "center": "center",
                        "width": "width",
                        "contrast": "contrast",
                        "offset": "offset",
                    }

                @property
                def frequency_parameters(self) -> list[str]:
                    return ["center"]

                def func(self, x: NDArray, parameters: NDArray) -> NDArray:
                    from qdmpy.fitting.models import _ensure_2d

                    parameters = _ensure_2d(parameters)
                    center = parameters[:, 0:1]
                    width_sq = parameters[:, 1:2] ** 2
                    contrast = parameters[:, 2:3]
                    offset = parameters[:, 3:4]
                    return 1 + offset - contrast * width_sq / ((x - center) ** 2 + width_sq)

        data = make_synthetic_odmr_data(
            shape=(4, 4), n_freq=40, model_name="ESRSINGLE", noise=0.0005, seed=5
        )
        fm = FitManager(model_name=name, settings=MOCK_SETTINGS, backend=TorchBackend(device="cpu"))
        result = fm.fit(data.data, data.frequencies)
        assert result.model_name == name
        assert np.all(result.parameters["states"] == 0)

    def test_numpy_coercing_model_raises_clear_error(self) -> None:
        class _NumpyOnly(Model):
            name: ClassVar[str] = "_QEP069_NUMPY_ONLY"

            def __init__(self) -> None:
                super().__init__("_QEP069_NUMPY_ONLY", 1, ["center", "width", "contrast", "offset"])
                self.model_id = -1

            @property
            def parameter_types(self) -> dict[str, str]:
                return {
                    "center": "center",
                    "width": "width",
                    "contrast": "contrast",
                    "offset": "offset",
                }

            @property
            def frequency_parameters(self) -> list[str]:
                return ["center"]

            def func(self, x: NDArray, parameters: NDArray) -> NDArray:
                parameters = np.atleast_2d(np.asarray(parameters))  # silently coerces
                center = parameters[:, 0:1]
                return np.exp(-((np.asarray(x) - center) ** 2))

        model = _NumpyOnly()
        _model, x, y, _true, x0 = _synthetic_esrsingle(2)
        constraints, ctypes = _free_constraints(2, 4)
        with pytest.raises(DependencyError, match="scipy"):
            TorchBackend(device="cpu").fit(
                y, x, x0, constraints, ctypes, model, FitBackendOptions(estimator="LSE")
            )


@pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch (gpu extra)")
class TestFitManagerIntegration:
    """TorchBackend through the full FitManager path."""

    def test_full_pipeline_all_models(self) -> None:
        for model_name in ("ESRSINGLE", "ESR15N", "ESR14N"):
            data = make_synthetic_odmr_data(
                shape=(4, 4), n_freq=50, model_name=model_name, noise=0.0005, seed=7
            )
            fm = FitManager(model_name=model_name, backend=TorchBackend(device="cpu"))
            result = fm.fit(data.data, data.frequencies)
            states = result.parameters["states"]
            assert np.mean(states == 0) == 1.0, model_name

    def test_settings_backend_torch(self) -> None:
        settings = QDMpySettings(fit=FitSettings(backend="torch"))
        fm = FitManager(model_name="ESRSINGLE", settings=settings)
        assert fm._backend.name == "torch"


def test_import_qdmpy_does_not_import_torch() -> None:
    """Torch import costs seconds; importing qdmpy must never pay it."""
    code = "import qdmpy, sys; sys.exit(1 if 'torch' in sys.modules else 0)"
    proc = subprocess.run(  # noqa: S603 — fixed argv, no untrusted input
        [sys.executable, "-c", code], check=False, capture_output=True
    )
    assert proc.returncode == 0, proc.stderr.decode()


def test_bounds_from_constraints_shared_helper() -> None:
    """The extracted module-level helper honors constraint types."""
    constraints = np.tile([1.0, 2.0, 3.0, 4.0], (3, 1))
    ctypes = np.array([1, 2], dtype=np.int32)  # LOWER, UPPER
    lower, upper = bounds_from_constraints(constraints, ctypes, 2)
    assert np.all(lower[:, 0] == 1.0)
    assert np.all(upper[:, 0] == np.inf)
    assert np.all(lower[:, 1] == -np.inf)
    assert np.all(upper[:, 1] == 4.0)
