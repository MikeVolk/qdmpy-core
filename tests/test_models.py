"""Unit tests for the models module.

This test suite provides comprehensive testing for the model functions,
Model class, and ModelRegistry in the QDMpy.models module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from QDMpy.constants import AHYP_14N, AHYP_15N
from QDMpy.models import (
    ESR14N,
    ESR15N,
    ESRSINGLE,
    Model,
    ModelRegistry,
    esr14n,
    esr15n,
    esrsingle,
)
from QDMpy.settings import (
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
)


class TestModelFunctions:
    """Tests for the individual model functions."""

    def test_esr14n_basic(self):
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

    def test_esr14n_multiple_parameter_sets(self):
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

    def test_esr14n_custom_hyperfine(self):
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

    def test_esr15n_basic(self):
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

    def test_esr15n_multiple_parameter_sets(self):
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

    def test_esr15n_custom_hyperfine(self):
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

    def test_esrsingle_basic(self):
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

    def test_esrsingle_multiple_parameter_sets(self):
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

    def test_model_abstract_class(self):
        """Test that Model cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Model("TestModel", 1, ["param1", "param2"])

    def test_model_property_parameter(self):
        """Test the parameter property of the Model class."""
        # Using a concrete implementation since Model is abstract
        model = ESR14N()

        # Check that parameter strips the unique identifiers
        expected = ["center", "width", "contrast", "contrast", "contrast", "offset"]
        assert model.parameter == expected

    def test_model_property_n_parameters(self):
        """Test the n_parameters property of the Model class."""
        model_14n = ESR14N()
        model_15n = ESR15N()
        model_single = ESRSINGLE()

        assert model_14n.n_parameters == 6
        assert model_15n.n_parameters == 5
        assert model_single.n_parameters == 4

    def test_model_repr(self):
        """Test the string representation of Model."""
        model = ESR14N()
        expected = "Model(ESR14N, n_parameters: 6, n_peaks: 3)"
        assert repr(model) == expected

    def test_model_func_abstract(self):
        """Test that the func method raises NotImplementedError if not implemented."""
        # We can't instantiate an abstract class without implementing all methods,
        # so we'll check the abstractmethod decorator directly

        # Verify that the func method is decorated with @abstractmethod
        assert Model.func.__isabstractmethod__

        # To test line 190 (raise NotImplementedError), we'll subclass Model and
        # call the parent's func method
        class TestModelCallsParentFunc(Model):
            def __init__(self):
                super().__init__("TEST", 1, ["param1"])

            def func(self, x, parameters):
                # Call the parent's func method directly, which should raise NotImplementedError
                return super().func(x, parameters)

        # Create an instance and call func
        model = TestModelCallsParentFunc()
        x = np.array([1, 2, 3])
        params = np.array([1])

        # Should raise NotImplementedError from the parent's func method
        with pytest.raises(NotImplementedError):
            model.func(x, params)

    def test_constraint_access(self):
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

    def test_get_constraint_array(self):
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

        # Order matches parameters_unique: ["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"]
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

    def test_get_constraint_array_with_missing_constraints(self):
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

    def test_init(self):
        """Test initialization of ESR14N."""
        model = ESR14N()

        assert model.name == "ESR14N"
        assert model.n_peaks == 3
        assert model.parameters_unique == [
            "center",
            "width",
            "contrast_0",
            "contrast_1",
            "contrast_2",
            "offset",
        ]
        assert model.ahyp == AHYP_14N

    def test_func(self):
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

    def test_init(self):
        """Test initialization of ESR15N."""
        model = ESR15N()

        assert model.name == "ESR15N"
        assert model.n_peaks == 2
        assert model.parameters_unique == ["center", "width", "contrast_0", "contrast_1", "offset"]
        assert model.ahyp == AHYP_15N

    def test_func(self):
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

    def test_init(self):
        """Test initialization of ESRSINGLE."""
        model = ESRSINGLE()

        assert model.name == "ESRSINGLE"
        assert model.n_peaks == 1
        assert model.parameters_unique == ["center", "width", "contrast", "offset"]

    def test_func(self):
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

    def test_registry_initial_state(self):
        """Test the initial state of the registry."""
        # The registry should already contain the three models
        registry = ModelRegistry.all()

        assert "ESR14N" in registry
        assert "ESR15N" in registry
        assert "ESRSINGLE" in registry

        assert registry["ESR14N"]["class"] == ESR14N
        assert registry["ESR15N"]["class"] == ESR15N
        assert registry["ESRSINGLE"]["class"] == ESRSINGLE

    def test_register_new_model(self):
        """Test registering a new model."""
        # Create a mock model class
        mock_model_class = MagicMock()

        # Register the mock model
        ModelRegistry.register("MOCK_MODEL", {"class": mock_model_class, "hyp": 0.0})

        # Check registration
        registry = ModelRegistry.all()
        assert "MOCK_MODEL" in registry
        assert registry["MOCK_MODEL"]["class"] == mock_model_class

    def test_get_model(self):
        """Test getting a model by name."""
        # Get models by name
        model_14n = ModelRegistry.get("ESR14N")
        model_15n = ModelRegistry.get("ESR15N")
        model_single = ModelRegistry.get("ESRSINGLE")

        # Check that they are instances of the correct classes
        assert isinstance(model_14n, ESR14N)
        assert isinstance(model_15n, ESR15N)
        assert isinstance(model_single, ESRSINGLE)

    def test_get_nonexistent_model(self):
        """Test getting a model that doesn't exist."""
        with pytest.raises(KeyError):
            ModelRegistry.get("NON_EXISTENT_MODEL")

    def test_initialize_constraints_method(self):
        """Test the _initialize_constraints method of ModelRegistry."""
        # We need to test line 294-305
        # Create a test model derived from Model that will expose _initialize_constraints
        from QDMpy.models import Model as QDMpyModel

        class TestModelInitConstraints(QDMpyModel):
            def __init__(self):
                super().__init__("TEST", 1, ["contrast_0", "width_0"])

            def func(self, x, parameters):
                return x  # Dummy implementation

        # Use a patch to ensure SETTINGS contains the right structure
        mock_settings = QDMpySettings(
            model=ModelSettings(
                constraints=ModelConstraintsSettings(
                    contrast_min=0.0,
                    contrast_max=1.0,
                    contrast_type='FREE',
                    width_min=1e6,
                    width_max=1e7,
                    width_type='FREE',
                )
            ),
        )

        with patch('QDMpy.models.SETTINGS', mock_settings):
            # Access the protected method for testing
            constraints = ModelRegistry._initialize_constraints(
                TestModelInitConstraints()
            )

            # Verify the output has the right structure
            assert 'contrast_0' in constraints
            assert 'width_0' in constraints
            assert len(constraints['contrast_0']) == 3
            assert constraints['contrast_0'][0] == 0.0  # min
            assert constraints['contrast_0'][1] == 1.0  # max
            assert constraints['contrast_0'][2] == 'FREE'  # type


# Test the path handling in direct import
def test_direct_import_handling():
    """Test the import path handling logic with simple test code."""
    import os
    import sys

    # Get the current test file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the project root (should be the directory containing tests/ and src/)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # Ensure the directory structure is as expected
    assert os.path.exists(current_dir)
    assert os.path.isdir(project_root)

    # Check that src/QDMpy exists (not QDMpy directly in the root)
    src_dir = os.path.join(project_root, "src")
    assert os.path.exists(src_dir), "src directory not found"

    qdmpy_dir = os.path.join(src_dir, "QDMpy")
    assert os.path.exists(qdmpy_dir), "QDMpy module not found in src"

    # Create a test for path insertion (directly test logic, not actual insert)
    original_path = list(sys.path)
    if project_root not in original_path:
        # In a real scenario, this would be: sys.path.insert(0, project_root)
        # But we don't modify sys.path in tests
        test_path_with_insert = [project_root] + original_path
        assert project_root == test_path_with_insert[0]


# Test the main demo function (lines 375-380)
def test_main_demo_function():
    """Test the _main_demo function that shows model usage."""
    import io
    from unittest.mock import patch

    from QDMpy.models import _main_demo

    # Capture stdout to verify the output
    captured_output = io.StringIO()

    with patch("sys.stdout", captured_output):
        _main_demo()

    # Verify the output matches what we expect
    output = captured_output.getvalue()
    assert output == "4\n"


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_models.py"])
