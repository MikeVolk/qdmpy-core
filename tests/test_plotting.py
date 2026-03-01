from __future__ import annotations

import importlib.util
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest


@unittest.skipIf(
    importlib.util.find_spec("qdmpy") is None,
    "qdmpy module not found - skipping plotting tests",
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
        from qdmpy.plotting import plot_fit_result_field_map

        # Should not raise any exceptions
        plot_fit_result_field_map(self.mock_result)

        # Check that calculate_b_field was called
        self.mock_result.calculate_b_field.assert_called()

    def test_plot_fit_result_field_map_with_save(self) -> None:
        """Test plot_fit_result_field_map with save option."""
        import os
        import tempfile

        from qdmpy.plotting import plot_fit_result_field_map

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_field_map.png")

            # Test with custom filename
            plot_fit_result_field_map(self.mock_result, save=True, filename=filename)

            # File should exist
            assert os.path.exists(filename)

    def test_plot_fit_result_field_map_basic_call(self) -> None:
        """Test plot_fit_result_field_map basic invocation."""
        from qdmpy.plotting import plot_fit_result_field_map

        plot_fit_result_field_map(self.mock_result)

        self.mock_result.calculate_b_field.assert_called()

    def test_plot_fit_result_parameter_map(self) -> None:
        """Test plot_fit_result_parameter_map function."""
        from qdmpy.plotting import plot_fit_result_parameter_map

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

        from qdmpy.plotting import plot_fit_result_parameter_map

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_param_map.png")

            plot_fit_result_parameter_map(self.mock_result, "center", save=True, filename=filename)

            # File should exist
            assert os.path.exists(filename)

    def test_plot_fit_result_parameter_map_basic_call(self) -> None:
        """Test plot_fit_result_parameter_map basic invocation."""
        from qdmpy.plotting import plot_fit_result_parameter_map

        plot_fit_result_parameter_map(self.mock_result, "center")

        self.mock_result.get_parameter_map.assert_called_with("center")

    def test_plot_fit_result_overview(self) -> None:
        """Test plot_fit_result_overview function."""
        from qdmpy.plotting import plot_fit_result_overview

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

        from qdmpy.plotting import plot_fit_result_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_overview.png")

            plot_fit_result_overview(self.mock_result, save=True, filename=filename)

            # File should exist
            assert os.path.exists(filename)

    def test_plot_fit_result_overview_limited_parameters(self) -> None:
        """Test plot_fit_result_overview with limited available parameters."""
        from qdmpy.plotting import plot_fit_result_overview

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
        from qdmpy.fitting.result import FitResult
        from qdmpy.plotting import (
            plot_fit_result_field_map,
            plot_fit_result_overview,
            plot_fit_result_parameter_map,
        )

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
        from qdmpy.plotting import plot_fit_result_parameter_map

        # Test with invalid parameter name
        def mock_get_parameter_map_error(param_name):
            if param_name == "invalid_param":
                raise KeyError(f"Parameter '{param_name}' not found")
            return np.random.random((10, 10))

        self.mock_result.get_parameter_map.side_effect = mock_get_parameter_map_error

        # Should propagate the KeyError
        with pytest.raises(KeyError):
            plot_fit_result_parameter_map(self.mock_result, "invalid_param")


# ---------------------------------------------------------------------------
# Folding diagnostic plot smoke tests
# ---------------------------------------------------------------------------


def _make_synthetic_folded_odmr():
    """Build a minimal FoldedODMR for plot smoke tests."""
    import xarray as xr

    from qdmpy.odmr.folding import FoldedODMR, FoldingSettings

    n_pol, ny, nx, n_df = 2, 4, 4, 20
    n_steps = 51
    pol_labels = ["neg", "pos"]
    delta_f = np.linspace(0.002, 0.015, n_df)

    folded = xr.DataArray(
        np.random.default_rng(0).random((n_pol, ny, nx, n_df)),
        dims=("polarity", "y", "x", "freq_idx"),
        coords={"polarity": pol_labels, "delta_f_ghz": ("freq_idx", delta_f)},
    )
    anti = xr.DataArray(
        np.random.default_rng(1).random((n_pol, ny, nx, n_df)) * 0.01,
        dims=("polarity", "y", "x", "freq_idx"),
        coords={"polarity": pol_labels, "delta_f_ghz": ("freq_idx", delta_f)},
    )
    d_zfs_map = xr.DataArray(
        np.full((n_pol, ny, nx), 2.870) + np.random.default_rng(2).random((n_pol, ny, nx)) * 0.001,
        dims=("polarity", "y", "x"),
        coords={"polarity": pol_labels},
    )
    fold_residual = xr.DataArray(
        np.random.default_rng(3).random((n_pol, ny, nx)) * 0.1,
        dims=("polarity", "y", "x"),
        coords={"polarity": pol_labels},
    )
    d_candidates = np.linspace(2.865, 2.875, n_steps)
    search_residual = np.random.default_rng(4).random((n_pol, n_steps)) * 0.01
    # Create a clear minimum
    search_residual[:, n_steps // 2] = 0.0001

    return FoldedODMR(
        folded_spectrum=folded,
        antisymmetric_spectrum=anti,
        d_zfs_map=d_zfs_map,
        fold_residual=fold_residual,
        settings=FoldingSettings(),
        d_candidates=d_candidates,
        search_residual=search_residual,
    )


class TestFoldingPlots:
    """Smoke tests for folding diagnostic plot functions."""

    def setup_method(self) -> None:
        self._original_show = plt.show
        plt.show = lambda: None  # suppress display

    def teardown_method(self) -> None:
        plt.close("all")
        plt.show = self._original_show

    def test_plot_folding_search_landscape(self) -> None:
        from qdmpy.plotting import plot_folding_search_landscape

        folded = _make_synthetic_folded_odmr()
        plot_folding_search_landscape(folded)

    def test_plot_folding_mean_spectrum(self) -> None:
        from qdmpy.plotting import plot_folding_mean_spectrum

        folded = _make_synthetic_folded_odmr()
        plot_folding_mean_spectrum(folded)

    def test_plot_folding_overview(self) -> None:
        from qdmpy.plotting import plot_folding_overview

        folded = _make_synthetic_folded_odmr()
        plot_folding_overview(folded)

    def test_folded_odmr_plot_method(self) -> None:
        """FoldedODMR.plot() calls plot_folding_overview without error."""
        folded = _make_synthetic_folded_odmr()
        folded.plot()

    def test_search_landscape_no_data(self) -> None:
        """plot_folding_search_landscape with None diagnostics returns early."""
        from qdmpy.plotting import plot_folding_search_landscape

        folded = _make_synthetic_folded_odmr()
        # Create a new instance without diagnostics
        from qdmpy.odmr.folding import FoldedODMR

        no_diag = FoldedODMR(
            folded_spectrum=folded.folded_spectrum,
            antisymmetric_spectrum=folded.antisymmetric_spectrum,
            d_zfs_map=folded.d_zfs_map,
            fold_residual=folded.fold_residual,
            settings=folded.settings,
        )
        plot_folding_search_landscape(no_diag)  # should not crash
