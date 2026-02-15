"""Tests for FitResult class and related functionality."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from QDMpy.constants import D_ZFS, GAMMA_NV
from QDMpy.result import FitResult


class TestFitResult:
    """Test suite for the FitResult class."""

    @pytest.fixture
    def sample_parameters(self):
        """Create sample fit parameters for testing."""
        n_pixels = 100
        return {
            "center": np.random.normal(2.87, 0.001, n_pixels),  # ~2.87 GHz
            "width_0": np.random.normal(0.0005, 0.00001, n_pixels),  # ~0.5 MHz in GHz
            "contrast": np.random.uniform(0.01, 0.1, n_pixels),  # 1-10% contrast
            "offset": np.random.normal(0, 0.01, n_pixels),  # Small offsets
            "chi2": np.random.exponential(1.0, n_pixels),  # Chi-squared values
            "states": np.random.choice([0, 1], n_pixels, p=[0.9, 0.1]),  # 90% convergence
        }

    @pytest.fixture
    def sample_fit_result(self, sample_parameters):
        """Create a sample FitResult for testing."""
        return FitResult(
            parameters=sample_parameters,
            scan_dimensions=(10, 10),
            pixel_spacing=4e-6,
            model_name="ESR15N",
            metadata={"test": True, "quality_metrics": {"mean_chi2": 1.0}},
        )

    def test_init_with_required_parameters(self, sample_parameters) -> None:
        """Test FitResult initialization with required parameters."""
        result = FitResult(
            parameters=sample_parameters,
            scan_dimensions=(10, 10),
            pixel_spacing=4e-6,
            model_name="ESR15N",
        )

        assert result.parameters == sample_parameters
        assert result.scan_dimensions == (10, 10)
        assert result.pixel_spacing == 4e-6
        assert result.model_name == "ESR15N"
        assert result.metadata == {}

    def test_init_with_optional_metadata(self, sample_parameters) -> None:
        """Test FitResult initialization with optional metadata."""
        metadata = {"test": True, "quality": 0.95}
        result = FitResult(
            parameters=sample_parameters,
            scan_dimensions=(10, 10),
            pixel_spacing=4e-6,
            model_name="ESR15N",
            metadata=metadata,
        )

        assert result.metadata == metadata

    def test_repr_string_representation(self, sample_fit_result) -> None:
        """Test string representation of FitResult."""
        repr_str = repr(sample_fit_result)
        assert "FitResult" in repr_str
        assert "ESR15N" in repr_str
        assert "n_pixels=100" in repr_str
        assert "parameters=6" in repr_str

    def test_centers_property(self, sample_fit_result) -> None:
        """Test centers property access."""
        centers = sample_fit_result.centers
        assert len(centers) == 100
        assert isinstance(centers, np.ndarray)
        np.testing.assert_array_equal(centers, sample_fit_result.parameters["center"])

    def test_linewidths_property_width_0(self, sample_fit_result) -> None:
        """Test linewidths property with width_0 parameter."""
        linewidths = sample_fit_result.linewidths
        assert len(linewidths) == 100
        np.testing.assert_array_equal(linewidths, sample_fit_result.parameters["width_0"])

    def test_linewidths_property_width_fallback(self, sample_parameters) -> None:
        """Test linewidths property fallback to width parameter."""
        # Remove width_0 and add width
        params = sample_parameters.copy()
        del params["width_0"]
        params["width"] = np.random.normal(0.0005, 0.00001, 100)

        result = FitResult(
            parameters=params, scan_dimensions=(10, 10), pixel_spacing=4e-6, model_name="ESRSINGLE"
        )

        linewidths = result.linewidths
        np.testing.assert_array_equal(linewidths, params["width"])

    def test_linewidths_property_no_width_error(self, sample_parameters) -> None:
        """Test linewidths property raises error when no width parameter exists."""
        # Remove all width parameters
        params = sample_parameters.copy()
        del params["width_0"]

        result = FitResult(
            parameters=params, scan_dimensions=(10, 10), pixel_spacing=4e-6, model_name="ESR15N"
        )

        with pytest.raises(KeyError, match="No linewidth parameter found"):
            _ = result.linewidths

    def test_contrasts_property(self, sample_fit_result) -> None:
        """Test contrasts property access."""
        contrasts = sample_fit_result.contrasts
        assert len(contrasts) == 100
        np.testing.assert_array_equal(contrasts, sample_fit_result.parameters["contrast"])

    def test_offsets_property(self, sample_fit_result) -> None:
        """Test offsets property access."""
        offsets = sample_fit_result.offsets
        assert len(offsets) == 100
        np.testing.assert_array_equal(offsets, sample_fit_result.parameters["offset"])

    def test_offsets_property_default_fallback(self, sample_parameters) -> None:
        """Test offsets property default fallback when offset not available."""
        # Remove offset parameter
        params = sample_parameters.copy()
        del params["offset"]

        result = FitResult(
            parameters=params, scan_dimensions=(10, 10), pixel_spacing=4e-6, model_name="ESR15N"
        )

        offsets = result.offsets
        assert len(offsets) == 100
        np.testing.assert_array_equal(offsets, np.zeros_like(result.centers))

    def test_chi2_property(self, sample_fit_result) -> None:
        """Test chi2 property access."""
        chi2 = sample_fit_result.chi2
        assert len(chi2) == 100
        np.testing.assert_array_equal(chi2, sample_fit_result.parameters["chi2"])

    def test_fit_states_property(self, sample_fit_result) -> None:
        """Test fit_states property access."""
        states = sample_fit_result.fit_states
        assert len(states) == 100
        np.testing.assert_array_equal(states, sample_fit_result.parameters["states"])

    def test_fit_states_property_default_fallback(self, sample_parameters) -> None:
        """Test fit_states property default fallback when states not available."""
        # Remove states parameter
        params = sample_parameters.copy()
        del params["states"]

        result = FitResult(
            parameters=params, scan_dimensions=(10, 10), pixel_spacing=4e-6, model_name="ESR15N"
        )

        states = result.fit_states
        assert len(states) == 100
        assert states.dtype == int
        np.testing.assert_array_equal(states, np.zeros_like(result.centers, dtype=int))

    def test_get_parameter_valid_param(self, sample_fit_result) -> None:
        """Test get_parameter with valid parameter name."""
        centers = sample_fit_result.get_parameter("center")
        np.testing.assert_array_equal(centers, sample_fit_result.parameters["center"])

    def test_get_parameter_invalid_param(self, sample_fit_result) -> None:
        """Test get_parameter with invalid parameter name."""
        with pytest.raises(KeyError, match="Parameter 'nonexistent' not found"):
            sample_fit_result.get_parameter("nonexistent")

    def test_get_parameter_map_reshaping(self, sample_fit_result) -> None:
        """Test get_parameter_map reshapes data correctly."""
        center_map = sample_fit_result.get_parameter_map("center")
        assert center_map.shape == (10, 10)

        # Verify reshaping is correct
        centers_flat = sample_fit_result.get_parameter("center")
        np.testing.assert_array_equal(center_map.flatten(), centers_flat)

    def test_calculate_b_field_first_time(self, sample_fit_result) -> None:
        """Test magnetic field calculation on first call."""
        b_field = sample_fit_result.calculate_b_field()

        assert b_field.shape == (10, 10)
        assert isinstance(b_field, np.ndarray)
        assert np.all(b_field >= 0)  # B-field should be positive

        # Check cache is populated
        assert sample_fit_result._b_field_cache is not None
        np.testing.assert_array_equal(b_field, sample_fit_result._b_field_cache)

    def test_calculate_b_field_cached(self, sample_fit_result) -> None:
        """Test magnetic field calculation uses cache on subsequent calls."""
        # First call
        b_field1 = sample_fit_result.calculate_b_field()

        # Second call should return cached result
        b_field2 = sample_fit_result.calculate_b_field()

        np.testing.assert_array_equal(b_field1, b_field2)
        assert b_field1 is b_field2  # Should be the same object

    def test_calculate_b_field_force_recalculate(self, sample_fit_result) -> None:
        """Test magnetic field calculation with force_recalculate=True."""
        # First call
        b_field1 = sample_fit_result.calculate_b_field()

        # Force recalculation
        b_field2 = sample_fit_result.calculate_b_field(force_recalculate=True)

        # Results should be the same but different objects
        np.testing.assert_array_equal(b_field1, b_field2)
        assert b_field1 is not b_field2

    def test_compute_b_field_calculations(self, sample_fit_result) -> None:
        """Test internal B-field calculation logic."""
        b_field = sample_fit_result._compute_b_field()

        # Get the expected calculation (all in GHz)
        centers_map = sample_fit_result.get_parameter_map("center")
        expected = np.abs(centers_map - D_ZFS) / GAMMA_NV

        np.testing.assert_array_almost_equal(b_field, expected)

    def test_get_fit_quality_metrics_complete(self, sample_fit_result) -> None:
        """Test fit quality metrics calculation with complete data."""
        metrics = sample_fit_result.get_fit_quality_metrics()

        # Check required metrics
        assert "mean_chi2" in metrics
        assert "median_chi2" in metrics
        assert "std_chi2" in metrics
        assert "n_pixels" in metrics
        assert "convergence_rate" in metrics
        assert "n_converged" in metrics

        # Check values are reasonable
        assert isinstance(metrics["mean_chi2"], float)
        assert isinstance(metrics["n_pixels"], int)
        assert metrics["n_pixels"] == 100
        assert 0 <= metrics["convergence_rate"] <= 1

    def test_get_fit_quality_metrics_missing_states(self, sample_parameters) -> None:
        """Test fit quality metrics without states parameter."""
        # Remove states parameter
        params = sample_parameters.copy()
        del params["states"]

        result = FitResult(
            parameters=params, scan_dimensions=(10, 10), pixel_spacing=4e-6, model_name="ESR15N"
        )

        metrics = result.get_fit_quality_metrics()

        # Should have basic metrics but not convergence rate
        assert "mean_chi2" in metrics
        assert "n_pixels" in metrics
        assert "convergence_rate" not in metrics
        assert "n_converged" not in metrics

    def test_get_fit_quality_metrics_with_metadata(self, sample_fit_result) -> None:
        """Test fit quality metrics includes metadata."""
        metrics = sample_fit_result.get_fit_quality_metrics()

        # Should include pre-computed metrics from metadata
        assert "mean_chi2" in metrics  # From calculation
        assert metrics["mean_chi2"] == 1.0  # From metadata override

    def test_save_results_to_file(self, sample_fit_result) -> None:
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.npz"

            sample_fit_result.save_results(filepath)

            assert filepath.exists()

            # Load and verify content
            data = np.load(filepath, allow_pickle=True)
            assert "model_name" in data
            assert "scan_dimensions" in data
            assert "parameters" in data

    def test_load_results_from_file(self, sample_fit_result) -> None:
        """Test loading results from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.npz"

            # Save first
            sample_fit_result.save_results(filepath)

            # Load
            loaded_data = FitResult.load_results(filepath)

            assert loaded_data["model_name"] == "ESR15N"
            assert tuple(loaded_data["scan_dimensions"]) == (10, 10)

    def test_load_results_file_not_found(self) -> None:
        """Test loading results from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Results file not found"):
            FitResult.load_results("nonexistent_file.npz")

    def test_save_load_roundtrip(self, sample_fit_result) -> None:
        """Test complete save/load roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "roundtrip_test.npz"

            # Calculate B-field to populate cache
            original_b_field = sample_fit_result.calculate_b_field()

            # Save
            sample_fit_result.save_results(filepath)

            # Load
            loaded_data = FitResult.load_results(filepath)

            # Verify key data preserved
            assert loaded_data["model_name"] == sample_fit_result.model_name
            assert tuple(loaded_data["scan_dimensions"]) == sample_fit_result.scan_dimensions
            assert loaded_data["pixel_spacing"] == sample_fit_result.pixel_spacing

            # Verify B-field was saved
            assert "b_field" in loaded_data
            np.testing.assert_array_equal(loaded_data["b_field"], original_b_field)
