"""Tests for FitResult class and related functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from QDMpy.constants import D_ZFS, GAMMA_NV
from QDMpy.exceptions import DataLoadError, DataShapeError, DataValidationError, ParameterError
from QDMpy.fitting.result import FitResult


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

        with pytest.raises(ParameterError, match="No linewidth parameter found"):
            _ = result.linewidths

    def test_contrasts_property(self, sample_fit_result) -> None:
        """Test contrasts property access via plain 'contrast' key (ESRSINGLE)."""
        contrasts = sample_fit_result.contrasts
        assert len(contrasts) == 100
        np.testing.assert_array_equal(contrasts, sample_fit_result.parameters["contrast"])

    def test_contrasts_property_contrast_0(self, sample_parameters) -> None:
        """Test contrasts falls back to 'contrast_0' for multi-dip models (ESR14N)."""
        params = {k: v for k, v in sample_parameters.items() if k != "contrast"}
        params["contrast_0"] = np.random.uniform(0.01, 0.1, 100)
        params["contrast_1"] = np.random.uniform(0.01, 0.1, 100)
        params["contrast_2"] = np.random.uniform(0.01, 0.1, 100)
        result = FitResult(parameters=params, scan_dimensions=(10, 10), pixel_spacing=4e-6, model_name="ESR14N")
        np.testing.assert_array_equal(result.contrasts, params["contrast_0"])

    def test_contrasts_property_raises_when_missing(self, sample_parameters) -> None:
        """Test contrasts raises ParameterError when no contrast key exists."""
        from QDMpy.exceptions import ParameterError
        params = {k: v for k, v in sample_parameters.items() if not k.startswith("contrast")}
        result = FitResult(parameters=params, scan_dimensions=(10, 10), pixel_spacing=4e-6, model_name="ESR14N")
        with pytest.raises(ParameterError):
            _ = result.contrasts

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
        with pytest.raises(ParameterError, match="Parameter 'nonexistent' not found"):
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
            loaded_result = FitResult.load_results(filepath)

            assert loaded_result.model_name == "ESR15N"
            assert loaded_result.scan_dimensions == (10, 10)

    def test_load_results_file_not_found(self) -> None:
        """Test loading results from non-existent file."""
        with pytest.raises(DataLoadError, match="Results file not found"):
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
            loaded_result = FitResult.load_results(filepath)

            # Verify key data preserved
            assert loaded_result.model_name == sample_fit_result.model_name
            assert loaded_result.scan_dimensions == sample_fit_result.scan_dimensions
            assert loaded_result.pixel_spacing == sample_fit_result.pixel_spacing

            # Verify parameters were preserved
            assert "center" in loaded_result.parameters
            # Note: b_field is not part of loaded FitResult parameters, would need separate loading if needed


class TestResolveSpatialDims:
    """Tests for FitResult._resolve_spatial_dims."""

    def _make_result(self, scan_dims: tuple[int, int]) -> FitResult:
        n = scan_dims[0] * scan_dims[1]
        return FitResult(
            parameters={"center": np.zeros(n), "chi2": np.zeros(n)},
            scan_dimensions=scan_dims,
            pixel_spacing=4e-6,
            model_name="ESR15N",
        )

    def test_matching_pixel_count(self) -> None:
        result = self._make_result((10, 10))
        assert result._resolve_spatial_dims(100) == (10, 10)

    def test_mismatched_square(self) -> None:
        result = self._make_result((10, 10))
        h, w = result._resolve_spatial_dims(36)
        assert h * w == 36
        assert h == 6
        assert w == 6

    def test_rectangular_aspect_ratio(self) -> None:
        result = self._make_result((10, 20))
        h, w = result._resolve_spatial_dims(50)
        assert h * w == 50
        assert w / h >= 1.0

    def test_prime_pixel_count(self) -> None:
        result = self._make_result((10, 10))
        h, w = result._resolve_spatial_dims(17)
        assert h * w == 17
        assert (h, w) == (1, 17) or (h, w) == (17, 1)


class TestNormalizeResonanceShape:
    """Tests for FitResult._normalize_resonance_shape."""

    def _make_result(self) -> FitResult:
        return FitResult(
            parameters={"center": np.zeros(100), "chi2": np.zeros(100)},
            scan_dimensions=(10, 10),
            pixel_spacing=4e-6,
            model_name="ESR15N",
        )

    def test_4d_input(self) -> None:
        result = self._make_result()
        arr = np.ones((2, 2, 50, 1))
        res, n_pol, n_frange, n_pix = result._normalize_resonance_shape(arr)
        assert res.shape == (2, 2, 50)
        assert n_pol == 2
        assert n_frange == 2
        assert n_pix == 50

    def test_3d_input(self) -> None:
        result = self._make_result()
        arr = np.ones((2, 2, 50))
        res, n_pol, n_frange, n_pix = result._normalize_resonance_shape(arr)
        assert res.shape == (2, 2, 50)
        assert n_pol == 2
        assert n_frange == 2
        assert n_pix == 50

    def test_2d_input(self) -> None:
        result = self._make_result()
        arr = np.ones((4, 50))
        res, n_pol, n_frange, n_pix = result._normalize_resonance_shape(arr)
        assert res.shape == (2, 2, 50)
        assert n_pol == 2
        assert n_frange == 2
        assert n_pix == 50

    def test_invalid_shape_raises(self) -> None:
        result = self._make_result()
        arr = np.ones((50,))
        with pytest.raises(DataShapeError, match="Unexpected center parameter shape"):
            result._normalize_resonance_shape(arr)


class TestCalcDeltaFromSingleCenter:
    """Tests for FitResult._calc_delta_from_single_center."""

    def _make_result(self) -> FitResult:
        return FitResult(
            parameters={"center": np.zeros(4), "chi2": np.zeros(4)},
            scan_dimensions=(2, 2),
            pixel_spacing=4e-6,
            model_name="ESR15N",
        )

    def test_two_frequency_ranges(self) -> None:
        result = self._make_result()
        resonance = np.array(
            [
                [[2.85, 2.85, 2.85, 2.85], [2.89, 2.89, 2.89, 2.89]],
                [[2.85, 2.85, 2.85, 2.85], [2.89, 2.89, 2.89, 2.89]],
            ]
        )
        delta = result._calc_delta_from_single_center(resonance, 2, 2, 2, 2)
        assert delta.shape == (2, 2, 2)  # (n_pol, H, W) — sign axis eliminated
        expected_diff = 0.04 / 2 / GAMMA_NV * 1e6
        np.testing.assert_allclose(delta[0], -expected_diff, rtol=1e-10)  # pol_0 (neg)
        np.testing.assert_allclose(delta[1], expected_diff, rtol=1e-10)  # pol_1 (pos)

    def test_single_frequency_range(self) -> None:
        result = self._make_result()
        freq = D_ZFS + 0.01
        resonance = np.array(
            [
                [[freq, freq, freq, freq]],
                [[freq, freq, freq, freq]],
            ]
        )
        delta = result._calc_delta_from_single_center(resonance, 2, 1, 2, 2)
        assert delta.shape == (2, 2, 2)  # (n_pol, H, W)
        expected_shift = 0.01 / GAMMA_NV * 1e6
        np.testing.assert_allclose(delta[0], -expected_shift, rtol=1e-10)  # pol_0 (neg)
        np.testing.assert_allclose(delta[1], expected_shift, rtol=1e-10)  # pol_1 (pos)


class TestComputeDeltaResonanceOrchestrator:
    """Integration tests for the rewritten _compute_delta_resonance."""

    def test_3d_center_two_franges(self) -> None:
        import xarray as xr

        center = np.array(
            [
                [[2.85, 2.85, 2.85, 2.85], [2.89, 2.89, 2.89, 2.89]],
                [[2.85, 2.85, 2.85, 2.85], [2.89, 2.89, 2.89, 2.89]],
            ]
        )
        result = FitResult(
            parameters={"center": center, "chi2": np.zeros(4)},
            scan_dimensions=(2, 2),
            pixel_spacing=4e-6,
            model_name="ESR15N",
        )
        delta = result._compute_delta_resonance()
        assert isinstance(delta, xr.DataArray)
        assert delta.dims == ("polarity", "y", "x")
        assert delta.shape == (2, 2, 2)
        assert list(delta.coords["polarity"].values) == ["neg", "pos"]



class TestFitResultValidation:
    """Tests for FitResult Pydantic validation."""

    def test_rejects_negative_pixel_spacing(self) -> None:
        """Test that negative pixel_spacing raises ValidationError."""
        with pytest.raises((DataValidationError, Exception)):
            FitResult(
                parameters={"center": np.ones(10), "chi2": np.ones(10)},
                scan_dimensions=(2, 5),
                pixel_spacing=-1.0,
                model_name="ESR15N",
            )

    def test_rejects_zero_pixel_spacing(self) -> None:
        """Test that zero pixel_spacing raises ValidationError."""
        with pytest.raises((DataValidationError, Exception)):
            FitResult(
                parameters={"center": np.ones(10), "chi2": np.ones(10)},
                scan_dimensions=(2, 5),
                pixel_spacing=0.0,
                model_name="ESR15N",
            )

    def test_rejects_zero_scan_dimensions(self) -> None:
        """Test that zero scan dimensions raise DataValidationError."""
        with pytest.raises((DataValidationError, Exception)):
            FitResult(
                parameters={"center": np.ones(10), "chi2": np.ones(10)},
                scan_dimensions=(0, 5),
                pixel_spacing=4e-6,
                model_name="ESR15N",
            )

    def test_rejects_negative_scan_dimensions(self) -> None:
        """Test that negative scan dimensions raise DataValidationError."""
        with pytest.raises((DataValidationError, Exception)):
            FitResult(
                parameters={"center": np.ones(10), "chi2": np.ones(10)},
                scan_dimensions=(-1, 5),
                pixel_spacing=4e-6,
                model_name="ESR15N",
            )

    def test_rejects_empty_parameters(self) -> None:
        """Test that empty parameters dict raises DataValidationError."""
        with pytest.raises((DataValidationError, Exception)):
            FitResult(
                parameters={},
                scan_dimensions=(2, 5),
                pixel_spacing=4e-6,
                model_name="ESR15N",
            )

    def test_is_pydantic_basemodel(self) -> None:
        """Test that FitResult is a Pydantic BaseModel instance."""
        from pydantic import BaseModel

        result = FitResult(
            parameters={"center": np.ones(10), "chi2": np.ones(10)},
            scan_dimensions=(2, 5),
            pixel_spacing=4e-6,
            model_name="ESR15N",
        )
        assert isinstance(result, BaseModel)
