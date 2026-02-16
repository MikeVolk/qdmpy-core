from __future__ import annotations

import importlib.util
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest


@unittest.skipIf(
    importlib.util.find_spec("QDMpy._core") is None,
    "QDMpy._core module not found - skipping plotting tests",
)
class TestPlotting(unittest.TestCase):
    """Tests for plotting functions in QDMpy."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create sample data for testing
        self.frequencies = np.linspace(2.87, 2.9, 100)
        self.data = 1.0 - 0.1 * np.exp(-(((self.frequencies - 2.885) / 0.01) ** 2))

        # Create a mock figure for testing
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Save original plt.figure method to restore later
        self.original_figure = plt.figure

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Close all figures
        plt.close("all")

        # Restore original plt.figure method
        plt.figure = self.original_figure

    def test_plotting_placeholder(self) -> None:
        """Placeholder test to ensure the test module runs."""
        # This test doesn't actually test anything, it's just here to make sure
        # the test module can be loaded and executed
        assert True


class TestFitResultPlotting(unittest.TestCase):
    """Tests for FitResult plotting functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock FitResult
        from unittest.mock import Mock

        import numpy as np

        self.mock_result = Mock()
        self.mock_result.scan_dimensions = (10, 10)
        self.mock_result.pixel_spacing = 4e-6
        self.mock_result.model_name = "ESR15N"

        # Mock B-field calculation
        self.b_field_data = np.random.uniform(0, 0.01, (10, 10))  # Tesla
        self.mock_result.calculate_b_field.return_value = self.b_field_data

        # Mock parameter maps
        self.center_data = np.random.normal(2.87, 0.001, (10, 10))
        self.width_data = np.random.normal(0.0005, 0.00001, (10, 10))
        self.contrast_data = np.random.uniform(0.01, 0.1, (10, 10))

        def mock_get_parameter_map(param_name):
            if param_name == "center":
                return self.center_data
            if param_name == "width_0":
                return self.width_data
            if param_name == "contrast":
                return self.contrast_data
            return np.random.random((10, 10))

        self.mock_result.get_parameter_map.side_effect = mock_get_parameter_map

        # Mock parameters dict for overview plot
        self.mock_result.parameters = {
            "center": self.center_data.flatten(),
            "width_0": self.width_data.flatten(),
            "contrast": self.contrast_data.flatten(),
            "chi2": np.random.exponential(1.0, 100),
        }

        # Save original show function to restore later
        self.original_show = plt.show
        plt.show = lambda: None  # Disable showing plots during tests

    def tearDown(self) -> None:
        """Clean up after tests."""
        plt.close("all")
        plt.show = self.original_show

    def test_plot_fit_result_field_map(self) -> None:
        """Test plot_fit_result_field_map function."""
        from QDMpy.plotting import plot_fit_result_field_map

        # Should not raise any exceptions
        plot_fit_result_field_map(self.mock_result)

        # Check that calculate_b_field was called
        self.mock_result.calculate_b_field.assert_called()

    def test_plot_fit_result_field_map_with_save(self) -> None:
        """Test plot_fit_result_field_map with save option."""
        import os
        import tempfile

        from QDMpy.plotting import plot_fit_result_field_map

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_field_map.png")

            # Test with custom filename
            plot_fit_result_field_map(self.mock_result, save=True, filename=filename)

            # File should exist
            assert os.path.exists(filename)

    def test_plot_fit_result_field_map_basic_call(self) -> None:
        """Test plot_fit_result_field_map basic invocation."""
        from QDMpy.plotting import plot_fit_result_field_map

        plot_fit_result_field_map(self.mock_result)

        self.mock_result.calculate_b_field.assert_called()

    def test_plot_fit_result_parameter_map(self) -> None:
        """Test plot_fit_result_parameter_map function."""
        from QDMpy.plotting import plot_fit_result_parameter_map

        # Test with different parameters
        plot_fit_result_parameter_map(self.mock_result, "center")
        self.mock_result.get_parameter_map.assert_called_with("center")

        plot_fit_result_parameter_map(self.mock_result, "width_0")
        self.mock_result.get_parameter_map.assert_called_with("width_0")

        plot_fit_result_parameter_map(self.mock_result, "contrast")
        self.mock_result.get_parameter_map.assert_called_with("contrast")

    def test_plot_fit_result_parameter_map_with_save(self) -> None:
        """Test plot_fit_result_parameter_map with save option."""
        import os
        import tempfile

        from QDMpy.plotting import plot_fit_result_parameter_map

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_param_map.png")

            plot_fit_result_parameter_map(self.mock_result, "center", save=True, filename=filename)

            # File should exist
            assert os.path.exists(filename)

    def test_plot_fit_result_parameter_map_basic_call(self) -> None:
        """Test plot_fit_result_parameter_map basic invocation."""
        from QDMpy.plotting import plot_fit_result_parameter_map

        plot_fit_result_parameter_map(self.mock_result, "center")

        self.mock_result.get_parameter_map.assert_called_with("center")

    def test_plot_fit_result_overview(self) -> None:
        """Test plot_fit_result_overview function."""
        from QDMpy.plotting import plot_fit_result_overview

        # Should not raise any exceptions
        plot_fit_result_overview(self.mock_result)

        # Should call both B-field calculation and parameter maps
        self.mock_result.calculate_b_field.assert_called()

        # Should call get_parameter_map for available parameters
        expected_calls = ["center", "width_0", "contrast", "chi2"]
        for param in expected_calls:
            self.mock_result.get_parameter_map.assert_any_call(param)

    def test_plot_fit_result_overview_with_save(self) -> None:
        """Test plot_fit_result_overview with save option."""
        import os
        import tempfile

        from QDMpy.plotting import plot_fit_result_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_overview.png")

            plot_fit_result_overview(self.mock_result, save=True, filename=filename)

            # File should exist
            assert os.path.exists(filename)

    def test_plot_fit_result_overview_limited_parameters(self) -> None:
        """Test plot_fit_result_overview with limited available parameters."""
        from QDMpy.plotting import plot_fit_result_overview

        # Modify mock to have fewer parameters
        self.mock_result.parameters = {
            "center": self.center_data.flatten(),
            "contrast": self.contrast_data.flatten(),
        }

        # Should still work with fewer parameters
        plot_fit_result_overview(self.mock_result)

        # Should still call B-field calculation
        self.mock_result.calculate_b_field.assert_called()

    def test_plotting_functions_with_real_fitresult(self) -> None:
        """Test plotting functions with a real FitResult object."""
        from QDMpy.plotting import (
            plot_fit_result_field_map,
            plot_fit_result_overview,
            plot_fit_result_parameter_map,
        )
        from QDMpy.result import FitResult

        # Create a real FitResult object
        n_pixels = 100
        parameters = {
            "center": np.random.normal(2.87, 0.001, n_pixels),
            "width_0": np.random.normal(0.0005, 0.00001, n_pixels),
            "contrast": np.random.uniform(0.01, 0.1, n_pixels),
            "offset": np.random.normal(0, 0.01, n_pixels),
            "chi2": np.random.exponential(1.0, n_pixels),
            "states": np.random.choice([0, 1], n_pixels, p=[0.9, 0.1]),
        }

        result = FitResult(
            parameters=parameters,
            scan_dimensions=(10, 10),
            pixel_spacing=4e-6,
            model_name="ESR15N",
            metadata={"test": True},
        )

        # All plotting functions should work without errors
        plot_fit_result_field_map(result)
        plot_fit_result_parameter_map(result, "center")
        plot_fit_result_parameter_map(result, "width_0")
        plot_fit_result_parameter_map(result, "contrast")
        plot_fit_result_overview(result)

    def test_plotting_error_handling(self) -> None:
        """Test error handling in plotting functions."""
        from QDMpy.plotting import plot_fit_result_parameter_map

        # Test with invalid parameter name
        def mock_get_parameter_map_error(param_name):
            if param_name == "invalid_param":
                raise KeyError(f"Parameter '{param_name}' not found")
            return np.random.random((10, 10))

        self.mock_result.get_parameter_map.side_effect = mock_get_parameter_map_error

        # Should propagate the KeyError
        with pytest.raises(KeyError):
            plot_fit_result_parameter_map(self.mock_result, "invalid_param")
