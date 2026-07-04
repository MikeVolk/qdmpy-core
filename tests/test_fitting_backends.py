"""Unit tests for the fit backend seam (QEP-068).

Covers backend resolution, the GpufitBackend/ScipyBackend contract, and the
end-to-end custom-model story that motivated the seam: a pure-Python model
(model_id=-1) can now be fitted via backend='scipy', which was previously
impossible (GpufitBackend hard-required pygpufit for every model).
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from qdmpy.exceptions import ParameterError
from qdmpy.fitting.backends import (
    GpufitBackend,
    ScipyBackend,
    resolve_backend,
    with_forced_availability,
)
from qdmpy.fitting.manager import FitManager
from qdmpy.fitting.models import ESRSINGLE, Model, ModelRegistry
from qdmpy.settings import FitSettings, QDMpySettings
from qdmpy.testing import FakeFitBackend, make_synthetic_odmr_data


class TestResolveBackend:
    """Backend name/instance resolution matrix."""

    def test_auto_resolves_to_gpufit(self) -> None:
        assert isinstance(resolve_backend("auto"), GpufitBackend)

    def test_none_resolves_to_gpufit(self) -> None:
        assert isinstance(resolve_backend(None), GpufitBackend)

    def test_gpufit_name_resolves_to_gpufit(self) -> None:
        assert isinstance(resolve_backend("gpufit"), GpufitBackend)

    def test_scipy_name_resolves_to_scipy(self) -> None:
        assert isinstance(resolve_backend("scipy"), ScipyBackend)

    def test_unknown_name_raises_parameter_error(self) -> None:
        with pytest.raises(ParameterError, match="Unknown fit backend"):
            resolve_backend("not_a_backend")

    def test_instance_passthrough(self) -> None:
        backend = FakeFitBackend()
        assert resolve_backend(backend) is backend

    def test_auto_never_raises_at_resolution_time(self) -> None:
        """Resolution never touches availability (config/execution split)."""
        with patch("qdmpy.settings.is_pygpufit_available", return_value=False):
            backend = resolve_backend("auto")
            assert isinstance(backend, GpufitBackend)
            assert backend.is_available() is False


class TestGpufitBackendSupports:
    """GpufitBackend.supports() model_id dispatch."""

    def test_supports_registered_model(self) -> None:
        backend = GpufitBackend()
        assert backend.supports(ModelRegistry.get("ESRSINGLE")) is True

    def test_rejects_custom_cpu_only_model(self) -> None:
        backend = GpufitBackend()

        class _CustomModel(Model):
            name: ClassVar[str] = "_CUSTOM_TEST_ONLY"

            def __init__(self) -> None:
                super().__init__("_CUSTOM_TEST_ONLY", 1, ["center", "width", "contrast", "offset"])
                self.model_id = -1

            @property
            def parameter_types(self) -> dict[str, str]:
                return {"center": "center", "width": "width", "contrast": "contrast", "offset": "offset"}

            @property
            def frequency_parameters(self) -> list[str]:
                return ["center"]

            def func(self, x: NDArray, parameters: NDArray) -> NDArray:
                return np.zeros((np.atleast_2d(parameters).shape[0], len(x)))

        assert backend.supports(_CustomModel()) is False


class TestWithForcedAvailability:
    """Deprecated gpu_available override, implemented via backend wrapping."""

    def test_forces_unavailable(self) -> None:
        backend = with_forced_availability(GpufitBackend(), available=False)
        assert backend.is_available() is False
        assert backend.name == "gpufit"

    def test_forces_available_regardless_of_real_state(self) -> None:
        with patch("qdmpy.settings.is_pygpufit_available", return_value=False):
            backend = with_forced_availability(GpufitBackend(), available=True)
            assert backend.is_available() is True


class TestFitManagerDeprecatedGpuAvailable:
    """FitManager(gpu_available=...) still works but warns."""

    def test_emits_deprecation_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match="gpu_available"):
            FitManager(model_name="ESRSINGLE", gpu_available=True)

    def test_rejects_both_backend_and_gpu_available(self) -> None:
        with pytest.raises(ParameterError, match="either 'backend' or"):
            FitManager(model_name="ESRSINGLE", backend="scipy", gpu_available=True)


class TestScipyBackend:
    """Behavioral tests for the CPU fitting backend."""

    def test_is_always_available(self) -> None:
        assert ScipyBackend().is_available() is True

    def test_supports_any_model(self) -> None:
        assert ScipyBackend().supports(ESRSINGLE()) is True

    def test_mle_estimator_falls_back_to_lse(self) -> None:
        """ScipyBackend only supports LSE; MLE requests still complete (fallback)."""
        data = make_synthetic_odmr_data(shape=(1, 2), n_freq=20, model_name="ESRSINGLE", noise=0.0)
        fit = FitManager(
            model_name="ESRSINGLE",
            settings=QDMpySettings(fit=FitSettings(estimator="MLE")),
            backend="scipy",
        )
        result = fit.fit(data.data, data.frequencies)
        assert result.model_name == "ESRSINGLE"

    def test_fits_synthetic_esrsingle_data(self) -> None:
        """End-to-end: ScipyBackend recovers a reasonable fit with low chi2."""
        data = make_synthetic_odmr_data(
            shape=(2, 2), n_freq=30, model_name="ESRSINGLE", noise=0.0005, seed=1
        )
        fit = FitManager(model_name="ESRSINGLE", backend="scipy")
        result = fit.fit(data.data, data.frequencies)

        assert result.parameters["center"].shape == (2, 2, 2, 2)
        assert np.all(result.parameters["states"] == 0)
        assert result.parameters["chi2"].mean() < 1e-3

    def test_bounds_from_constraints_respects_constraint_type(self) -> None:
        """FREE columns are ignored even if the constraint array has finite values."""
        n_params = 2
        constraints = np.tile([1.0, 2.0, 3.0, 4.0], (5, 1))
        constraint_types = np.array([0, 3], dtype=np.int32)  # FREE, LOWER_UPPER

        lower, upper = ScipyBackend._bounds_from_constraints(constraints, constraint_types, n_params)

        assert np.all(lower[:, 0] == -np.inf)
        assert np.all(upper[:, 0] == np.inf)
        assert np.all(lower[:, 1] == 3.0)
        assert np.all(upper[:, 1] == 4.0)


class TestCustomPurePythonModel:
    """The custom-model contract documented on Model: model_id=-1 is CPU-only."""

    MODEL_NAME = "_QEP068_CUSTOM_TEST_MODEL"

    @classmethod
    def setup_class(cls) -> None:
        if cls.MODEL_NAME in ModelRegistry.all():
            return

        @ModelRegistry.register
        class _CustomModel(Model):
            name: ClassVar[str] = cls.MODEL_NAME

            def __init__(self) -> None:
                super().__init__(cls.MODEL_NAME, 1, ["center", "width", "contrast", "offset"])
                self.model_id = -1  # CPU-only; gpufit not used for custom models

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
                parameters = np.atleast_2d(parameters)
                center = parameters[:, 0:1]
                width_sq = parameters[:, 1:2] ** 2
                contrast = parameters[:, 2:3]
                offset = parameters[:, 3:4]
                dip = contrast * width_sq / ((x - center) ** 2 + width_sq)
                return 1 + offset - dip

    def test_fits_successfully_via_scipy_backend(self) -> None:
        data = make_synthetic_odmr_data(
            shape=(2, 2), n_freq=30, model_name="ESRSINGLE", noise=0.0005, seed=1
        )
        fit = FitManager(model_name=self.MODEL_NAME, backend="scipy")
        result = fit.fit(data.data, data.frequencies)

        assert result.model_name == self.MODEL_NAME
        assert result.parameters["center"].shape == (2, 2, 2, 2)

    def test_gpufit_backend_rejects_custom_model(self) -> None:
        """GpufitBackend.supports() is False for model_id=-1, independent of GPU availability."""
        backend = GpufitBackend()
        model = ModelRegistry.get(self.MODEL_NAME)
        assert backend.supports(model) is False


class TestFitSettingsBackendKnob:
    """FitSettings.backend default and explicit override."""

    def test_default_is_auto(self) -> None:
        assert FitSettings().backend == "auto"

    def test_settings_backend_used_when_no_explicit_backend_given(self) -> None:
        settings = QDMpySettings(fit=FitSettings(backend="scipy"))
        fit = FitManager(model_name="ESRSINGLE", settings=settings)
        assert fit._backend.name == "scipy"

    def test_explicit_backend_overrides_settings(self) -> None:
        settings = QDMpySettings(fit=FitSettings(backend="scipy"))
        fit = FitManager(model_name="ESRSINGLE", settings=settings, backend="gpufit")
        assert fit._backend.name == "gpufit"

    def test_rejects_unknown_backend_name_at_settings_construction(self) -> None:
        """The Literal type on FitSettings.backend rejects unknown names eagerly."""
        with pytest.raises(ValueError, match="backend"):
            FitSettings(backend="not_a_backend")
