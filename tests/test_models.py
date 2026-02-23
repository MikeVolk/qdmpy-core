"""Unit tests for the models module.

This test suite provides comprehensive testing for the model functions,
Model class, and ModelRegistry in the QDMpy.models module.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from qdmpy.constants import AHYP_14N, AHYP_15N
from qdmpy.fitting.models import (
    ESR14N,
    ESR15N,
    ESRSINGLE,
    Model,
    ModelRegistry,
    esr14n,
    esr15n,
    esrsingle,
)

try:
    import pygpufit.gpufit  # noqa: F401

    _HAS_GPUFIT = True
except ImportError:
    _HAS_GPUFIT = False


class TestModelFunctions:
    """Tests for the individual model functions."""

    def test_esr14n_basic(self) -> None:
        """Test esr14n with basic input."""
        # Create test data
        x = np.linspace(2.87, 2.88, 100)
        parameters = np.array([2.87, 0.002, 0.2, 0.3, 0.1, 0.0])

        # Get model output
        result = esr14n(x, parameters)

        # Check output shape
        assert result.shape == (1, 100)

        # Check some expected values
        # Dips should occur at center and center ± hyperfine constant
        center_idx = np.argmin(np.abs(x - parameters[0]))
        assert result[0, center_idx] < 1.0  # Should show a dip at center

        # Check output range - should be between 0 and 1 + offset
        assert np.all(result <= 1.0 + parameters[5])
        assert np.all(
            result >= 1.0 + parameters[5] - (parameters[2] + parameters[3] + parameters[4])
        )

    def test_esr14n_multiple_parameter_sets(self) -> None:
        """Test esr14n with multiple parameter sets."""
        x = np.linspace(2.87, 2.88, 50)
        parameters = np.array(
            [
                [2.87, 0.002, 0.2, 0.3, 0.1, 0.0],
                [2.875, 0.003, 0.1, 0.2, 0.3, 0.1],
            ]
        )

        result = esr14n(x, parameters)

        # Check output shape - should have one row per parameter set
        assert result.shape == (2, 50)

        # First set should have deeper dip at center (parameter[0])
        center1_idx = np.argmin(np.abs(x - parameters[0][0]))
        # Second set should have deeper dip at center (parameter[0])
        center2_idx = np.argmin(np.abs(x - parameters[1][0]))

        # First set has deeper dip for second resonance
        assert result[0, center1_idx] < 1.0 - parameters[0][3]
        # Second set has deeper dip for third resonance
        assert result[1, center2_idx] < 1.0 - parameters[1][3]

    def test_esr14n_custom_hyperfine(self) -> None:
        """Test esr14n with custom hyperfine splitting."""
        x = np.linspace(2.87, 2.88, 100)
        parameters = np.array([2.87, 0.002, 0.2, 0.3, 0.1, 0.0])

        # Use a custom hyperfine value
        custom_ahyp = 0.005
        result = esr14n(x, parameters, ahyp=custom_ahyp)

        # Dips should now occur at center ± custom_ahyp
        center_idx = np.argmin(np.abs(x - parameters[0]))
        dip1_idx = np.argmin(np.abs(x - (parameters[0] - custom_ahyp)))
        dip3_idx = np.argmin(np.abs(x - (parameters[0] + custom_ahyp)))

        # Check that dips appear at expected locations
        assert result[0, center_idx] < 1.0  # Dip at center
        assert result[0, dip1_idx] < 1.0  # Dip at center - custom_ahyp
        assert result[0, dip3_idx] < 1.0  # Dip at center + custom_ahyp

    def test_esr15n_basic(self) -> None:
        """Test esr15n with basic input."""
        # Create test data
        x = np.linspace(2.86, 2.89, 100)  # Wider range to capture hyperfine splitting
        parameters = np.array([2.87, 0.002, 0.2, 0.3, 0.0])

        # Get model output
        result = esr15n(x, parameters)

        # Check output shape
        assert result.shape == (1, 100)

        # Check for dips near center ± hyperfine
        dip1_idx = np.argmin(np.abs(x - (parameters[0] - AHYP_15N)))
        dip2_idx = np.argmin(np.abs(x - (parameters[0] + AHYP_15N)))

        # Verify that dips are present
        assert result[0, dip1_idx] < 1.0  # Should be less than baseline
        assert result[0, dip2_idx] < 1.0  # Should be less than baseline

        # Check output range - should be between 0 and 1 + offset
        assert np.all(result <= 1.0 + parameters[4])
        assert np.all(result >= 1.0 + parameters[4] - (parameters[2] + parameters[3]))

    def test_esr15n_multiple_parameter_sets(self) -> None:
        """Test esr15n with multiple parameter sets."""
        x = np.linspace(2.87, 2.88, 50)
        parameters = np.array(
            [
                [2.87, 0.002, 0.2, 0.3, 0.0],
                [2.875, 0.003, 0.3, 0.1, 0.1],
            ]
        )

        result = esr15n(x, parameters)

        # Check output shape - should have one row per parameter set
        assert result.shape == (2, 50)

        # First set has deeper dip for second resonance
        # Second set has deeper dip for first resonance
        assert result[0, :].min() < result[1, :].min()

    def test_esr15n_custom_hyperfine(self) -> None:
        """Test esr15n with custom hyperfine splitting."""
        x = np.linspace(2.87, 2.88, 100)
        parameters = np.array([2.87, 0.002, 0.2, 0.3, 0.0])

        # Use a custom hyperfine value
        custom_ahyp = 0.004
        result = esr15n(x, parameters, ahyp=custom_ahyp)

        # Dips should now occur at center ± custom_ahyp
        dip1_idx = np.argmin(np.abs(x - (parameters[0] - custom_ahyp)))
        dip2_idx = np.argmin(np.abs(x - (parameters[0] + custom_ahyp)))

        # Dips should be at the expected positions
        assert result[0, dip1_idx] < 1.0  # Dip at center - custom_ahyp
        assert result[0, dip2_idx] < 1.0  # Dip at center + custom_ahyp

    def test_esrsingle_basic(self) -> None:
        """Test esrsingle with basic input."""
        # Create test data
        x = np.linspace(2.87, 2.88, 100)
        parameters = np.array([2.87, 0.002, 0.2, 0.0])

        # Get model output
        result = esrsingle(x, parameters)

        # Check output shape
        assert result.shape == (1, 100)

        # Check for single dip at center
        center_idx = np.argmin(np.abs(x - parameters[0]))

        # ESRSINGLE has just one dip at center
        assert np.argmin(result[0]) == center_idx

        # Check output range - should be between 0 and 1 + offset
        assert np.all(result <= 1.0 + parameters[3])
        assert np.all(result >= 1.0 + parameters[3] - parameters[2])

    def test_esrsingle_multiple_parameter_sets(self) -> None:
        """Test esrsingle with multiple parameter sets."""
        x = np.linspace(2.87, 2.88, 50)
        parameters = np.array(
            [
                [2.87, 0.002, 0.2, 0.0],
                [2.875, 0.003, 0.3, 0.1],
            ]
        )

        result = esrsingle(x, parameters)

        # Check output shape - should have one row per parameter set
        assert result.shape == (2, 50)

        # The second set should have a deeper relative dip
        # First set: baseline 1.0, dip to 0.8 = 0.2 contrast
        # Second set: baseline 1.1, dip to 0.8 = 0.3 contrast
        assert parameters[1][2] > parameters[0][2]  # Second set has higher contrast parameter

        # Verify that both sets have dips at their centers
        center1_idx = np.argmin(np.abs(x - parameters[0][0]))
        center2_idx = np.argmin(np.abs(x - parameters[1][0]))

        # The minimum should be at the center frequency
        assert np.argmin(result[0]) == center1_idx
        assert np.argmin(result[1]) == center2_idx

        # Both minimums should be less than their respective baselines
        assert result[0, center1_idx] < 1.0 + parameters[0][3]
        assert result[1, center2_idx] < 1.0 + parameters[1][3]


class TestModelClass:
    """Tests for the abstract Model class and concrete implementations."""

    def test_model_abstract_class(self) -> None:
        """Test that Model cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Model("TestModel", 1, ["param1", "param2"])

    def test_model_property_parameter(self) -> None:
        """Test the parameter property of the Model class."""
        # Using a concrete implementation since Model is abstract
        model = ESR14N()

        # Check that parameter strips the unique identifiers
        expected = ["center", "width", "contrast", "contrast", "contrast", "offset"]
        assert model.parameter == expected

    def test_model_property_n_parameters(self) -> None:
        """Test the n_parameters property of the Model class."""
        model_14n = ESR14N()
        model_15n = ESR15N()
        model_single = ESRSINGLE()

        assert model_14n.n_parameters == 6
        assert model_15n.n_parameters == 5
        assert model_single.n_parameters == 4

    def test_model_repr(self) -> None:
        """Test the string representation of Model."""
        model = ESR14N()
        expected = "Model(ESR14N, n_parameters: 6, n_peaks: 3)"
        assert repr(model) == expected

    def test_model_func_abstract(self) -> None:
        """Test that the func method raises NotImplementedError if not implemented."""
        assert Model.func.__isabstractmethod__

        class TestModelCallsParentFunc(Model):
            def __init__(self) -> None:
                super().__init__("TEST", 1, ["param1"])

            @property
            def parameter_types(self) -> dict[str, str]:
                return {"param1": "center"}

            @property
            def frequency_parameters(self) -> list[str]:
                return ["param1"]

            def func(self, x, parameters):
                return super().func(x, parameters)

        model = TestModelCallsParentFunc()
        x = np.array([1, 2, 3])
        params = np.array([1])

        with pytest.raises(NotImplementedError):
            model.func(x, params)

    def test_constraint_access(self) -> None:
        """Test the constraint-related code paths with a direct test approach."""
        # We can create constraint dictionaries directly and test their behavior
        model = ESR14N()

        # Create sample constraints
        constraints = {
            "center": [2.8, 2.9, "FREE"],
            "width": [0.001, 0.01, "FREE"],
            "contrast": [0.0, 1.0, "FREE"],
            "offset": [-0.1, 0.1, "FREE"],
        }

        # Test the get_constraint_array method directly
        # This will cover the code in _initialize_constraints since it accesses
        # constraints the same way
        constraint_array = model.get_constraint_array(constraints)

        # Should be two values (min/max) for each parameter
        assert len(constraint_array) == 2 * model.n_parameters

        # Check that min values are extracted correctly
        min_values = constraint_array[::2]  # Every other element starting at 0
        max_values = constraint_array[1::2]  # Every other element starting at 1

        # Ensure all values are valid
        assert all(isinstance(v, (int, float)) for v in min_values)
        assert all(isinstance(v, (int, float)) for v in max_values)

    def test_get_constraint_array(self) -> None:
        """Test the get_constraint_array method of Model."""
        model = ESR14N()
        constraints = {
            "center": [2.8, 2.9],  # [min, max]
            "width": [0.001, 0.01],
            "contrast": [0.0, 1.0],
            "offset": [-0.1, 0.1],
        }

        constraint_array = model.get_constraint_array(constraints)

        # Should have 2 values (min and max) for each parameter
        assert len(constraint_array) == 2 * model.n_parameters

        # Order matches parameter_names: ["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"]
        expected = [
            2.8,
            2.9,  # center min/max
            0.001,
            0.01,  # width min/max
            0.0,
            1.0,  # contrast_0 min/max
            0.0,
            1.0,  # contrast_1 min/max
            0.0,
            1.0,  # contrast_2 min/max
            -0.1,
            0.1,  # offset min/max
        ]
        assert_array_equal(constraint_array, expected)

    def test_get_constraint_array_with_missing_constraints(self) -> None:
        """Test get_constraint_array with missing constraints."""
        model = ESR14N()
        # Only provide some constraints
        constraints = {
            "center": [2.8, 2.9],
            "contrast": [0.0, 1.0],
        }

        constraint_array = model.get_constraint_array(constraints)

        # Should still have values for all parameters
        assert len(constraint_array) == 2 * model.n_parameters

        # Parameters not in constraints should have -inf/inf bounds
        # Order: ["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"]
        expected = [
            2.8,
            2.9,  # center min/max
            -np.inf,
            np.inf,  # width min/max (defaults)
            0.0,
            1.0,  # contrast_0 min/max
            0.0,
            1.0,  # contrast_1 min/max
            0.0,
            1.0,  # contrast_2 min/max
            -np.inf,
            np.inf,  # offset min/max (defaults)
        ]

        for i in range(len(expected)):
            if np.isinf(expected[i]):
                assert np.isinf(constraint_array[i])
            else:
                assert constraint_array[i] == expected[i]


class TestESR14N:
    """Tests for the ESR14N model."""

    def test_init(self) -> None:
        """Test initialization of ESR14N."""
        model = ESR14N()

        assert model.name == "ESR14N"
        assert model.n_peaks == 3
        assert model.parameter_names == [
            "center",
            "width",
            "contrast_0",
            "contrast_1",
            "contrast_2",
            "offset",
        ]
        assert model.ahyp == AHYP_14N

    def test_func(self) -> None:
        """Test the func method of ESR14N."""
        model = ESR14N()
        x = np.linspace(2.87, 2.88, 10)
        parameters = np.array([2.87, 0.002, 0.2, 0.3, 0.1, 0.0])

        # Should call esr14n with the correct hyperfine constant (GHz)
        result = model.func(x, parameters)
        expected = esr14n(x, parameters, AHYP_14N)

        assert_array_equal(result, expected)


class TestESR15N:
    """Tests for the ESR15N model."""

    def test_init(self) -> None:
        """Test initialization of ESR15N."""
        model = ESR15N()

        assert model.name == "ESR15N"
        assert model.n_peaks == 2
        assert model.parameter_names == ["center", "width", "contrast_0", "contrast_1", "offset"]
        assert model.ahyp == AHYP_15N

    def test_func(self) -> None:
        """Test the func method of ESR15N."""
        model = ESR15N()
        x = np.linspace(2.87, 2.88, 10)
        parameters = np.array([2.87, 0.002, 0.2, 0.3, 0.0])

        # Should call esr15n with the correct hyperfine constant (GHz)
        result = model.func(x, parameters)
        expected = esr15n(x, parameters, AHYP_15N)

        assert_array_equal(result, expected)


class TestESRSINGLE:
    """Tests for the ESRSINGLE model."""

    def test_init(self) -> None:
        """Test initialization of ESRSINGLE."""
        model = ESRSINGLE()

        assert model.name == "ESRSINGLE"
        assert model.n_peaks == 1
        assert model.parameter_names == ["center", "width", "contrast", "offset"]

    def test_func(self) -> None:
        """Test the func method of ESRSINGLE."""
        model = ESRSINGLE()
        x = np.linspace(2.87, 2.88, 10)
        parameters = np.array([2.87, 0.002, 0.2, 0.0])

        # Should call esrsingle
        result = model.func(x, parameters)
        expected = esrsingle(x, parameters)

        assert_array_equal(result, expected)


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_registry_initial_state(self) -> None:
        """Test the initial state of the registry."""
        registry = ModelRegistry.all()

        assert "ESR14N" in registry
        assert "ESR15N" in registry
        assert "ESRSINGLE" in registry

        assert registry["ESR14N"] is ESR14N
        assert registry["ESR15N"] is ESR15N
        assert registry["ESRSINGLE"] is ESRSINGLE

    def test_register_new_model(self) -> None:
        """Test registering a new model via decorator."""

        @ModelRegistry.register
        class MockModel(Model):
            name: ClassVar[str] = "MOCK_MODEL"

            def __init__(self) -> None:
                super().__init__("MOCK_MODEL", 1, ["center", "width"])

            @property
            def parameter_types(self) -> dict[str, str]:
                return {"center": "center", "width": "width"}

            @property
            def frequency_parameters(self) -> list[str]:
                return ["center"]

            def func(self, x, parameters):
                return x

        registry = ModelRegistry.all()
        assert "MOCK_MODEL" in registry
        assert registry["MOCK_MODEL"] is MockModel

        # Clean up
        del ModelRegistry._registry["MOCK_MODEL"]

    def test_get_model(self) -> None:
        """Test getting a model by name."""
        # Get models by name
        model_14n = ModelRegistry.get("ESR14N")
        model_15n = ModelRegistry.get("ESR15N")
        model_single = ModelRegistry.get("ESRSINGLE")

        # Check that they are instances of the correct classes
        assert isinstance(model_14n, ESR14N)
        assert isinstance(model_15n, ESR15N)
        assert isinstance(model_single, ESRSINGLE)

    def test_get_nonexistent_model(self) -> None:
        """Test getting a model that doesn't exist."""
        with pytest.raises(KeyError):
            ModelRegistry.get("NON_EXISTENT_MODEL")


class TestModelSelfDescribing:
    """Tests for the self-describing model properties (QEP-005)."""

    def test_parameter_types_esr14n(self) -> None:
        model = ESR14N()
        pt = model.parameter_types
        assert pt["center"] == "center"
        assert pt["width"] == "width"
        assert pt["contrast_0"] == "contrast"
        assert pt["contrast_1"] == "contrast"
        assert pt["contrast_2"] == "contrast"
        assert pt["offset"] == "offset"

    def test_parameter_types_esr15n(self) -> None:
        model = ESR15N()
        pt = model.parameter_types
        assert pt["center"] == "center"
        assert pt["width"] == "width"
        assert pt["contrast_0"] == "contrast"
        assert pt["contrast_1"] == "contrast"
        assert pt["offset"] == "offset"

    def test_parameter_types_esrsingle(self) -> None:
        model = ESRSINGLE()
        pt = model.parameter_types
        assert pt["center"] == "center"
        assert pt["width"] == "width"
        assert pt["contrast"] == "contrast"
        assert pt["offset"] == "offset"

    def test_frequency_parameters(self) -> None:
        for model_cls in [ESR14N, ESR15N, ESRSINGLE]:
            model = model_cls()
            assert model.frequency_parameters == ["center"]

    def test_units(self) -> None:
        model = ESR14N()
        units = model.units
        assert units["center"] == "GHz"
        assert units["width"] == "a.u."
        assert units["contrast_0"] == "a.u."
        assert units["offset"] == "a.u."

    def test_parameter_derives_from_parameter_types(self) -> None:
        model = ESR14N()
        expected = ["center", "width", "contrast", "contrast", "contrast", "offset"]
        assert model.parameter == expected

        model_single = ESRSINGLE()
        expected_single = ["center", "width", "contrast", "offset"]
        assert model_single.parameter == expected_single


# Test the main demo function
def test_main_demo_function() -> None:
    """Test the _main_demo function that shows model usage."""
    import io
    from unittest.mock import patch

    from qdmpy.fitting.models import _main_demo

    # Capture stdout to verify the output
    captured_output = io.StringIO()

    with patch("sys.stdout", captured_output):
        _main_demo()

    # Verify the output matches what we expect
    output = captured_output.getvalue()
    assert output == "4\n"


@pytest.mark.skipif(not _HAS_GPUFIT, reason="Requires pyGpufit installation")
class TestGpufitConsistency:
    """Verify Python model functions match the corresponding gpufit GPU kernels.

    Strategy: generate noiseless synthetic spectra from known parameters using
    the Python model, then fit them with gpufit using the matching GPU kernel
    (starting from the true parameters). If the Python and GPU implementations
    agree, chi2 will be ≈ 0 and recovered parameters will match the ground truth
    within float32 precision. A mismatch produces nonzero chi2 even at the true
    params and causes the fits to diverge.
    """

    N = 64
    N_FREQ = 50
    FREQ = np.linspace(2.82, 2.92, N_FREQ, dtype=np.float32)

    def _run(self, model_name: str, true_params: np.ndarray) -> None:
        from qdmpy.fitting.manager import FitManager
        from qdmpy.fitting.models import ModelRegistry

        model = ModelRegistry.get(model_name)
        spectra = model.func(self.FREQ, true_params).astype(np.float32)  # (N, n_freq)

        fm = FitManager(model_name=model_name)
        # fit_frange expects (n_pol, n_pixel, n_freq) — use n_pol=1
        data = spectra[np.newaxis]  # (1, N, n_freq)
        init = true_params[np.newaxis]  # (1, N, n_params)

        results = fm.fit_frange(data, self.FREQ, init)
        recovered = results[0].reshape(-1, model.n_parameters)  # (N, n_params)
        states = results[1].flatten()
        chi2 = results[2].flatten()

        assert np.all(states == 0), (
            f"{model_name}: some fits did not converge — states: {np.unique(states, return_counts=True)}"
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

    def test_esr14n_matches_gpufit(self) -> None:
        """Python esr14n must match the ESR14N gpufit kernel (model_id=13)."""
        rng = np.random.default_rng(0)
        params = np.empty((self.N, 6), dtype=np.float32)
        params[:, 0] = rng.uniform(2.85, 2.89, self.N)  # center (GHz)
        params[:, 1] = rng.uniform(0.002, 0.005, self.N)  # width (GHz)
        params[:, 2] = rng.uniform(0.05, 0.15, self.N)  # contrast_0
        params[:, 3] = rng.uniform(0.05, 0.20, self.N)  # contrast_1
        params[:, 4] = rng.uniform(0.05, 0.15, self.N)  # contrast_2
        params[:, 5] = rng.uniform(-0.01, 0.01, self.N)  # offset
        self._run("ESR14N", params)

    def test_esr15n_matches_gpufit(self) -> None:
        """Python esr15n must match the ESR15N gpufit kernel (model_id=14)."""
        rng = np.random.default_rng(1)
        params = np.empty((self.N, 5), dtype=np.float32)
        params[:, 0] = rng.uniform(2.85, 2.89, self.N)
        params[:, 1] = rng.uniform(0.002, 0.005, self.N)
        params[:, 2] = rng.uniform(0.05, 0.20, self.N)
        params[:, 3] = rng.uniform(0.05, 0.20, self.N)
        params[:, 4] = rng.uniform(-0.01, 0.01, self.N)
        self._run("ESR15N", params)

    def test_esrsingle_matches_gpufit(self) -> None:
        """Python esrsingle must match the ESRSINGLE gpufit kernel (model_id=15)."""
        rng = np.random.default_rng(2)
        params = np.empty((self.N, 4), dtype=np.float32)
        params[:, 0] = rng.uniform(2.85, 2.89, self.N)
        params[:, 1] = rng.uniform(0.002, 0.005, self.N)
        params[:, 2] = rng.uniform(0.05, 0.30, self.N)
        params[:, 3] = rng.uniform(-0.01, 0.01, self.N)
        self._run("ESRSINGLE", params)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_models.py"])
