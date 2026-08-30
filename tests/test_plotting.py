from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from typing import Any, cast

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

    def test_parameter_map_center_colorbar_uses_ghz_label(self) -> None:
        """Center parameter colorbar label follows GHz internal frequency convention."""
        from qdmpy.plotting import plot_fit_result_parameter_map

        plot_fit_result_parameter_map(self.mock_result, "center")
        fig = plt.gcf()
        fig.canvas.draw()

        cbar_axes = [ax for ax in fig.axes if not ax.images]
        assert cbar_axes
        assert cbar_axes[0].get_ylabel() == "Resonance Center (GHz)"

    def test_field_map_colorbar_height_matches_image_axis(self) -> None:
        """Colorbar axis height tracks the corresponding image axis height."""
        from qdmpy.plotting import plot_fit_result_field_map

        plot_fit_result_field_map(self.mock_result)
        fig = plt.gcf()
        fig.canvas.draw()

        image_axes = [ax for ax in fig.axes if ax.images]
        cbar_axes = [ax for ax in fig.axes if not ax.images]
        assert len(image_axes) == 1
        assert len(cbar_axes) >= 1

        image_pos = image_axes[0].get_position()
        cbar_pos = cbar_axes[0].get_position()
        assert abs(image_pos.y0 - cbar_pos.y0) < 1e-4
        assert abs(image_pos.height - cbar_pos.height) < 1e-4

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

    def test_overview_suptitle_does_not_overlap_top_row(self) -> None:
        """Overview reserves top margin for suptitle in multi-panel layout."""
        from qdmpy.plotting import plot_fit_result_overview

        plot_fit_result_overview(self.mock_result)
        fig = plt.gcf()
        fig.canvas.draw()

        assert fig.texts
        suptitle_y = max(text.get_position()[1] for text in fig.texts)
        top_axis = max(ax.get_position().y1 for ax in fig.axes)
        assert suptitle_y > top_axis

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


# ---------------------------------------------------------------------------
# ODMR spectra / fluorescence / model-detection / magnetic-map plot tests
# ---------------------------------------------------------------------------


def _make_odmr_data():
    """Build a minimal ODMRData for plot smoke tests."""
    from qdmpy.odmr.data import ODMRData

    rng = np.random.default_rng(42)
    n_pol, n_frange, ny, nx, n_freq = 2, 2, 4, 4, 50
    n_pixels = ny * nx

    raw_4d = rng.random((n_pol, n_frange, n_pixels, n_freq)).astype(np.float64)
    # Give it a dip-like shape so peak detection works
    for p in range(n_pol):
        for f in range(n_frange):
            raw_4d[p, f] = 1.0 - 0.05 * np.exp(-(((np.arange(n_freq) - n_freq // 2) / 5.0) ** 2))

    freq_low = np.linspace(2.72e9, 2.87e9, n_freq)
    freq_high = np.linspace(2.87e9, 3.02e9, n_freq)
    frequencies = np.stack([freq_low, freq_high])  # (2, n_freq) in Hz

    return ODMRData.from_numpy(raw_4d, (ny, nx), frequencies)


class TestODMRSpectraPlot:
    """Smoke tests for plot_odmr_spectra."""

    def setup_method(self) -> None:
        self._original_show = plt.show
        plt.show = lambda: None

    def teardown_method(self) -> None:
        plt.close("all")
        plt.show = self._original_show

    def test_plot_odmr_spectra(self) -> None:
        from qdmpy.plotting import plot_odmr_spectra

        odmr_data = _make_odmr_data()
        plot_odmr_spectra(odmr_data, 0, 0)

    def test_odmr_manager_delegates(self) -> None:
        """ODMR.plot_spectra delegates to plotting.plot_odmr_spectra."""
        from qdmpy.odmr.manager import ODMR

        odmr_data = _make_odmr_data()
        odmr = ODMR(odmr_data)
        odmr.process_data()
        odmr.plot_spectra(0, 0)


class TestFluorescenceCorrectionPlot:
    """Smoke tests for plot_fluorescence_correction."""

    def setup_method(self) -> None:
        self._original_show = plt.show
        plt.show = lambda: None

    def teardown_method(self) -> None:
        plt.close("all")
        plt.show = self._original_show

    def test_plot_fluorescence_correction(self) -> None:
        from qdmpy.plotting import plot_fluorescence_correction

        odmr_data = _make_odmr_data()
        plot_fluorescence_correction(odmr_data, 0.2)

    def test_polarity_labels_match_polarity_coord(self) -> None:
        """Regression test: the subplot titles used to hardcode
        {0: "+", 1: "-"}, backwards from the actual polarity coord
        (index 0 is "neg", index 1 is "pos" -- pol_0/pol_1 convention).
        """
        from qdmpy.plotting import plot_fluorescence_correction

        odmr_data = _make_odmr_data()
        assert list(odmr_data.data.coords["polarity"].values[:2]) == ["neg", "pos"]

        plot_fluorescence_correction(odmr_data, 0.2)

        fig = plt.gcf()
        titles = [ax.get_title() for ax in fig.axes]
        neg_titles = [t for t in titles if t.startswith("Polarity: -")]
        pos_titles = [t for t in titles if t.startswith("Polarity: +")]
        assert neg_titles, titles
        assert pos_titles, titles

    def test_preview_delegates(self) -> None:
        """preview_fluorescence_correction delegates to plotting module."""
        from qdmpy.odmr.processors import preview_fluorescence_correction

        odmr_data = _make_odmr_data()
        preview_fluorescence_correction(odmr_data, 0.2)


class TestModelDetectionPlot:
    """Smoke tests for plot_model_detection."""

    def setup_method(self) -> None:
        self._original_show = plt.show
        plt.show = lambda: None

    def teardown_method(self) -> None:
        plt.close("all")
        plt.show = self._original_show

    def test_plot_model_detection_no_freq(self) -> None:
        from qdmpy.plotting import plot_model_detection

        rng = np.random.default_rng(7)
        spectra = 1.0 - 0.05 * rng.random((2, 2, 16, 50))
        plot_model_detection(spectra)

    def test_plot_model_detection_with_freq(self) -> None:
        from qdmpy.plotting import plot_model_detection

        rng = np.random.default_rng(7)
        spectra = 1.0 - 0.05 * rng.random((2, 2, 16, 50))
        freq = np.stack([np.linspace(2.72, 2.87, 50), np.linspace(2.87, 3.02, 50)])
        plot_model_detection(spectra, freq)

    def test_guess_module_delegates(self) -> None:
        """guess.plot_model_detection delegates to plotting module."""
        from qdmpy.fitting.guess import plot_model_detection

        rng = np.random.default_rng(7)
        spectra = 1.0 - 0.05 * rng.random((2, 2, 16, 50))
        plot_model_detection(spectra)


class TestMagneticComponentPlot:
    """Smoke tests for plot_magnetic_component."""

    def setup_method(self) -> None:
        self._original_show = plt.show
        plt.show = lambda: None

    def teardown_method(self) -> None:
        plt.close("all")
        plt.show = self._original_show

    def test_plot_magnetic_component(self) -> None:
        import xarray as xr

        from qdmpy.magnetic_map import MagneticMap
        from qdmpy.plotting import plot_magnetic_component

        rng = np.random.default_rng(99)
        ny, nx = 8, 8
        coords = {"y": np.arange(ny), "x": np.arange(nx)}
        attrs = {"pixel_spacing": 4e-6}

        def _da(name):
            return xr.DataArray(
                rng.random((ny, nx)),
                dims=("y", "x"),
                coords=coords,
                attrs={**attrs, "component": name},
            )

        mag = MagneticMap(
            b111=_da("b111"),
            bx=_da("Bx"),
            by=_da("By"),
            bz=_da("Bz"),
            btotal=_da("Btotal"),
            nv_axis=(0.0, 0.0, 1.0),
        )
        plot_magnetic_component(mag, "Bz")

    def test_display_delegates(self) -> None:
        """MagneticMap.display delegates to plotting module."""
        import xarray as xr

        from qdmpy.magnetic_map import MagneticMap

        rng = np.random.default_rng(99)
        ny, nx = 8, 8
        coords = {"y": np.arange(ny), "x": np.arange(nx)}
        attrs = {"pixel_spacing": 4e-6}

        def _da(name):
            return xr.DataArray(
                rng.random((ny, nx)),
                dims=("y", "x"),
                coords=coords,
                attrs={**attrs, "component": name},
            )

        mag = MagneticMap(
            b111=_da("b111"),
            bx=_da("Bx"),
            by=_da("By"),
            bz=_da("Bz"),
            btotal=_da("Btotal"),
            nv_axis=(0.0, 0.0, 1.0),
        )
        mag.display("Bz")

    def test_invalid_component_raises(self) -> None:
        import xarray as xr

        from qdmpy.magnetic_map import MagneticMap
        from qdmpy.plotting import plot_magnetic_component

        rng = np.random.default_rng(99)
        ny, nx = 4, 4
        coords = {"y": np.arange(ny), "x": np.arange(nx)}
        attrs = {"pixel_spacing": 4e-6}

        def _da(name):
            return xr.DataArray(
                rng.random((ny, nx)),
                dims=("y", "x"),
                coords=coords,
                attrs={**attrs, "component": name},
            )

        mag = MagneticMap(
            b111=_da("b111"),
            bx=_da("Bx"),
            by=_da("By"),
            bz=_da("Bz"),
            btotal=_da("Btotal"),
            nv_axis=(0.0, 0.0, 1.0),
        )
        with pytest.raises(ValueError, match="not in"):
            plot_magnetic_component(mag, "invalid")


class TestDisplayLayout:
    """Tests for qdm display figure-size heuristics."""

    def test_display_layout_uses_shorter_spectra_rows(self) -> None:
        from qdmpy.plotting.display import _compute_display_layout

        _figsize, row_heights = _compute_display_layout(
            height=1200,
            width=1920,
            has_images=True,
            spec_rows=2,
        )

        # [map, map, optical, spectra, spectra]
        assert len(row_heights) == 5
        assert row_heights[3] < row_heights[0]
        assert row_heights[4] < row_heights[0]

    def test_display_layout_scales_map_rows_with_aspect(self) -> None:
        from qdmpy.plotting.display import _compute_display_layout

        _wide_figsize, wide_rows = _compute_display_layout(
            height=1000,
            width=2000,
            has_images=False,
            spec_rows=0,
        )
        _tall_figsize, tall_rows = _compute_display_layout(
            height=2000,
            width=1000,
            has_images=False,
            spec_rows=0,
        )

        assert wide_rows[0] < tall_rows[0]


class TestB111MapPlot:
    """Tests for B111 component plotting styles."""

    def setup_method(self) -> None:
        self._original_show = plt.show
        plt.show = lambda: None

    def teardown_method(self) -> None:
        plt.close("all")
        plt.show = self._original_show

    def test_plot_b111_map_induced_uses_sequential_scaling(self) -> None:
        import xarray as xr

        from qdmpy.plotting import plot_b111_map

        rng = np.random.default_rng(123)
        induced = 500 + 1000 * rng.random((8, 8))
        remanent = rng.normal(0.0, 50.0, (8, 8))

        result = SimpleNamespace(
            pixel_spacing=4e-6,
            scan_dimensions=(8, 8),
            model_name="ESR15N",
            b111={
                "remanent": xr.DataArray(remanent),
                "induced": xr.DataArray(induced),
            },
        )

        plot_b111_map(cast(Any, result), component="induced")
        fig = plt.gcf()
        fig.canvas.draw()

        ax = next(axis for axis in fig.axes if axis.images)
        image = ax.images[0]
        assert image.get_cmap().name == "viridis"
        assert image.norm.vmin is not None
        assert image.norm.vmax is not None
        assert image.norm.vmin >= 0
        assert image.norm.vmin != -image.norm.vmax
