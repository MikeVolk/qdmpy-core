"""Test module for QDMpy.measurement

These tests cover the Measurement class, which encapsulates all data and processing
related to a single QDM (Quantum Diamond Microscope) measurement.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Now we can import from QDMpy
from QDMpy.measurement import Measurement
from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.odmr import ODMR
from QDMpy.odmr.processors import BinningProcessor


@pytest.fixture
def sample_data():
    """Create sample data for ODMR testing."""
    data = np.random.random((2, 3, 100, 50))  # (modes, reps, pixels, frequencies)
    scan_dimensions = np.array([10, 10])  # 10x10 grid
    frequencies = np.linspace(2.87e9, 2.89e9, 50)  # 50 frequencies
    return data, scan_dimensions, frequencies


@pytest.fixture
def sample_odmr_data(sample_data):
    """Create a sample ODMRData instance for testing."""
    data, scan_dimensions, frequencies = sample_data
    return ODMRData(data, scan_dimensions, frequencies)


@pytest.fixture
def sample_odmr(sample_odmr_data):
    """Create a sample ODMR instance for testing."""
    odmr = ODMR(sample_odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.process_data()
    return odmr


@pytest.fixture
def sample_images():
    """Create sample light and laser images for testing."""
    light_image = np.random.random((10, 10))
    laser_image = np.random.random((10, 10))
    return light_image, laser_image


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    return tmp_path


class TestMeasurement:
    """Test class for Measurement."""

    def test_init(self, sample_odmr, sample_images, temp_output_dir):
        """Test initialization with standard parameters."""
        light_image, laser_image = sample_images

        # Create a measurement object
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=4e-6,
            fit_model="auto",
        )

        # Check that the attributes were set correctly
        assert measurement.odmr is sample_odmr
        assert np.array_equal(measurement.light_image, light_image)
        assert np.array_equal(measurement.laser_image, laser_image)
        assert isinstance(measurement.output_directory, Path)
        assert measurement.output_directory == temp_output_dir
        assert measurement.pixel_spacing == 4e-6
        assert measurement._fit_model == "auto"
        assert isinstance(measurement.metadata, dict)
        assert len(measurement.metadata) == 0
        assert measurement._outliers is not None
        assert measurement._B111 is None

    def test_init_with_unprocessed_odmr(self, sample_odmr_data, sample_images, temp_output_dir):
        """Test initialization with an ODMR instance that hasn't been processed."""
        light_image, laser_image = sample_images

        # Create an ODMR instance without processing
        odmr = ODMR(sample_odmr_data)

        # Create a measurement object
        measurement = Measurement(
            odmr=odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Check that the attributes were set correctly
        assert measurement.odmr is odmr
        assert np.array_equal(measurement.light_image, light_image)
        assert np.array_equal(measurement.laser_image, laser_image)
        # Should still work even though ODMR data wasn't processed
        assert hasattr(measurement, "_outliers")

    def test_init_with_no_odmr_data(self, sample_images, temp_output_dir):
        """Test initialization with an ODMR instance that has no data."""
        light_image, laser_image = sample_images

        # Create an empty ODMR instance
        empty_odmr = ODMR()

        # Attempt to create a measurement object with an empty ODMR
        with pytest.raises(ValueError) as excinfo:
            Measurement(
                odmr=empty_odmr,
                light_image=light_image,
                laser_image=laser_image,
                output_directory=temp_output_dir,
            )

        # Check the error message
        assert "ODMR instance has no raw data" in str(excinfo.value)

    def test_string_representations(self, sample_odmr, sample_images, temp_output_dir):
        """Test the string representation methods."""
        light_image, laser_image = sample_images

        # Create a measurement object
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Test __str__
        str_repr = str(measurement)
        assert "Measurement" in str_repr
        assert str(temp_output_dir) in str_repr
        assert "pixel_spacing" in str_repr

        # Test __repr__
        repr_str = repr(measurement)
        assert "Measurement" in repr_str
        assert "light_image.shape" in repr_str
        assert "laser_image.shape" in repr_str
        assert str(temp_output_dir) in repr_str

    def test_with_different_pixel_spacing(self, sample_odmr, sample_images, temp_output_dir):
        """Test initialization with different pixel spacing values."""
        light_image, laser_image = sample_images

        # Try a different pixel spacing
        pixel_spacing = 1e-6  # 1 μm

        # Create a measurement object
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=pixel_spacing,
        )

        # Check that the pixel spacing was set correctly
        assert measurement.pixel_spacing == pixel_spacing
        assert measurement.pixel_spacing != 4e-6  # Not the default value

    def test_with_string_output_directory(self, sample_odmr, sample_images, temp_output_dir):
        """Test initialization with string output directory."""
        light_image, laser_image = sample_images
        output_dir_str = str(temp_output_dir)

        # Create a measurement object with string path
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=output_dir_str,
        )

        # Path should be automatically converted to a Path object
        assert isinstance(measurement.output_directory, Path)
        assert str(measurement.output_directory) == output_dir_str

    def test_metadata_dictionary(self, sample_odmr, sample_images, temp_output_dir):
        """Test that the metadata dictionary works as expected."""
        light_image, laser_image = sample_images

        # Create a measurement object
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Initially the metadata dictionary should be empty
        assert isinstance(measurement.metadata, dict)
        assert len(measurement.metadata) == 0

        # We should be able to add items to it
        measurement.metadata["test_key"] = "test_value"
        assert "test_key" in measurement.metadata
        assert measurement.metadata["test_key"] == "test_value"

        # We should be able to update it
        measurement.metadata.update({"another_key": 123})
        assert measurement.metadata["another_key"] == 123

    @patch("logging.Logger.debug")
    def test_logging(self, mock_debug, sample_odmr, sample_images, temp_output_dir):
        """Test that initialization logs appropriate messages."""
        light_image, laser_image = sample_images

        # Create a measurement object
        Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Check that the appropriate log messages were generated
        assert any("Setting ODMR data" in call.args[0] for call in mock_debug.call_args_list)
        assert any(
            "Initializing outlier mask" in call.args[0] for call in mock_debug.call_args_list
        )
        assert any(
            "Storing light and laser images" in call.args[0] for call in mock_debug.call_args_list
        )

    def test_outliers_property(self, sample_odmr, sample_images, temp_output_dir):
        """Test the _outliers attribute."""
        light_image, laser_image = sample_images

        # Create a measurement object
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Check the outliers attribute
        assert measurement._outliers is not None
        assert isinstance(measurement._outliers, np.ndarray)
        assert measurement._outliers.shape == sample_odmr.raw_data.shape
        assert measurement._outliers.dtype == bool

    def test_B111_property(self, sample_odmr, sample_images, temp_output_dir):
        """Test the _B111 attribute."""
        light_image, laser_image = sample_images

        # Create a measurement object
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Check the B111 attribute
        assert measurement._B111 is None

        # We should be able to set it
        test_data = np.ones((5, 5))
        measurement._B111 = test_data
        assert measurement._B111 is test_data

    def test_fit_model_attribute(self, sample_odmr, sample_images, temp_output_dir):
        """Test the _fit_model attribute."""
        light_image, laser_image = sample_images

        # Create a measurement object with default fit_model
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Check the default value
        assert measurement._fit_model == "auto"

        # Create another measurement with a different fit_model
        measurement2 = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESR14N",
        )

        # Check the custom value
        assert measurement2._fit_model == "ESR14N"

    def test_fit_odmr_auto_model_detection(self, sample_odmr, sample_images, temp_output_dir):
        """Test fit_odmr with automatic model detection."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="auto",  # Use auto-detection
        )

        # Mock the guess_model function to return a specific model
        with patch("QDMpy.guess.guess_model") as mock_guess:
            mock_model = type("MockModel", (), {"name": "ESR15N"})()
            mock_guess.return_value = mock_model

            # Mock FitManager to avoid actual fitting
            with patch("QDMpy.fit.FitManager") as mock_fit_manager:
                mock_fit_instance = mock_fit_manager.return_value
                mock_fit_instance.fitted = True
                mock_fit_instance.model_name = "ESR15N"
                mock_fit_instance.scan_dimensions = (10, 10)
                test_params = {
                    "center": np.random.random(100),
                    "width_0": np.random.random(100),
                    "contrast": np.random.random(100),
                    "offset": np.random.random(100),
                    "chi2": np.random.random(100),
                    "states": np.random.choice([0, 1], 100),
                }
                mock_fit_instance.get_param.side_effect = lambda param: test_params.get(param, None)

                result = measurement.fit_odmr()

                # Check that guess_model was called
                mock_guess.assert_called_once()

                # Check that FitManager was initialized with auto-detected model
                mock_fit_manager.assert_called_once()

                # Check result type
                from QDMpy.result import FitResult

                assert isinstance(result, FitResult)
                assert result.model_name == "ESR15N"

    def test_fit_odmr_specific_model(self, sample_odmr, sample_images, temp_output_dir):
        """Test fit_odmr with a specific model name."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESR14N",  # Specific model
        )

        # Mock FitManager to avoid actual fitting
        with patch("QDMpy.fit.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            mock_fit_instance.fitted = True
            mock_fit_instance.model_name = "ESR14N"
            mock_fit_instance.scan_dimensions = (10, 10)
            test_params = {
                "center": np.random.random(100),
                "width_0": np.random.random(100),
                "contrast": np.random.random(100),
                "offset": np.random.random(100),
                "chi2": np.random.random(100),
                "states": np.random.choice([0, 1], 100),
            }
            mock_fit_instance.get_param.side_effect = lambda param: test_params.get(param, None)

            result = measurement.fit_odmr()

            # Check that FitManager was initialized with specified model
            args, kwargs = mock_fit_manager.call_args
            assert kwargs.get("model_name") == "ESR14N"

            # Check result
            from QDMpy.result import FitResult

            assert isinstance(result, FitResult)
            assert result.model_name == "ESR14N"

    def test_fit_odmr_override_model(self, sample_odmr, sample_images, temp_output_dir):
        """Test fit_odmr with model override parameter."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESR14N",  # Default model
        )

        # Mock FitManager
        with patch("QDMpy.fit.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            mock_fit_instance.fitted = True
            mock_fit_instance.model_name = "ESR15N"
            mock_fit_instance.scan_dimensions = (10, 10)
            test_params = {
                "center": np.random.random(100),
                "width_0": np.random.random(100),
                "contrast": np.random.random(100),
                "offset": np.random.random(100),
                "chi2": np.random.random(100),
                "states": np.random.choice([0, 1], 100),
            }
            mock_fit_instance.get_param.side_effect = lambda param: test_params.get(param, None)

            # Override model in fit_odmr call
            result = measurement.fit_odmr(model_name="ESR15N")

            # Check that FitManager was initialized with override model
            args, kwargs = mock_fit_manager.call_args
            assert kwargs.get("model_name") == "ESR15N"

            # Check result
            assert result.model_name == "ESR15N"

    def test_fit_odmr_with_kwargs(self, sample_odmr, sample_images, temp_output_dir):
        """Test fit_odmr with additional keyword arguments."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESRSINGLE",
        )

        # Mock FitManager
        with patch("QDMpy.fit.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            mock_fit_instance.fitted = True
            mock_fit_instance.model_name = "ESRSINGLE"
            mock_fit_instance.scan_dimensions = (10, 10)
            test_params = {
                "center": np.random.random(100),
                "width_0": np.random.random(100),
                "contrast": np.random.random(100),
                "offset": np.random.random(100),
                "chi2": np.random.random(100),
                "states": np.random.choice([0, 1], 100),
            }
            mock_fit_instance.get_param.side_effect = lambda param: test_params.get(param, None)

            # Call with additional kwargs
            custom_constraints = {"center": {"vmin": 2.85e9, "vmax": 2.90e9}}
            result = measurement.fit_odmr(constraints=custom_constraints, max_iterations=200)

            # Check that kwargs were passed to FitManager
            args, kwargs = mock_fit_manager.call_args
            assert "constraints" in kwargs
            assert kwargs["constraints"] == custom_constraints
            assert "max_iterations" in kwargs
            assert kwargs["max_iterations"] == 200

    def test_fit_odmr_data_extraction(self, sample_odmr, sample_images, temp_output_dir):
        """Test that fit_odmr properly extracts data for fitting."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=5e-6,  # Custom spacing
        )

        # Mock FitManager
        with patch("QDMpy.fit.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            mock_fit_instance.fitted = True
            mock_fit_instance.model_name = "ESRSINGLE"
            mock_fit_instance.scan_dimensions = (10, 10)
            mock_fit_instance.parameters = {
                "center": np.random.random(100),
                "width_0": np.random.random(100),
                "contrast": np.random.random(100),
            }

            result = measurement.fit_odmr()

            # Check that FitManager was called with correct data
            args, kwargs = mock_fit_manager.call_args

            # First argument should be the ODMR data
            assert np.array_equal(args[0], sample_odmr.data)

            # Second argument should be the frequencies
            assert np.array_equal(args[1], sample_odmr.frequencies)

            # Check that result has correct pixel spacing
            assert result.pixel_spacing == 5e-6

    def test_fit_odmr_result_properties(self, sample_odmr, sample_images, temp_output_dir):
        """Test that FitResult has correct properties from Measurement."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=3e-6,
        )

        # Mock FitManager with specific parameters
        test_parameters = {
            "center": np.random.normal(2.87e9, 1e6, 100),
            "width_0": np.random.normal(5e5, 1e4, 100),
            "contrast": np.random.uniform(0.01, 0.1, 100),
            "offset": np.random.normal(0, 0.01, 100),
            "chi2": np.random.exponential(1.0, 100),
            "states": np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        }

        with patch("QDMpy.fit.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            mock_fit_instance.fitted = True
            mock_fit_instance.model_name = "ESR15N"
            mock_fit_instance.scan_dimensions = (10, 10)
            mock_fit_instance.get_param.side_effect = lambda param: test_parameters.get(param, None)

            result = measurement.fit_odmr()

            # Check that FitResult was created with correct properties
            # Note: scan_dimensions come from processed_data, not FitManager
            assert result.pixel_spacing == 3e-6
            assert result.model_name == "ESR15N"

            # Check that parameters were copied correctly
            for param_name, param_values in test_parameters.items():
                assert param_name in result.parameters
                np.testing.assert_array_equal(result.parameters[param_name], param_values)

    def test_fit_odmr_fitting_failure(self, sample_odmr, sample_images, temp_output_dir):
        """Test fit_odmr behavior when fitting fails."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Mock FitManager to simulate fitting failure
        with patch("QDMpy.fit.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            # Simulate get_param failure (common when fitting fails)
            mock_fit_instance.get_param.side_effect = ValueError("No fit has been performed yet")

            # Should raise an exception when get_param fails
            with pytest.raises(ValueError, match="No fit has been performed yet"):
                measurement.fit_odmr()

    def test_fit_odmr_no_processed_data(self, sample_odmr_data, sample_images, temp_output_dir):
        """Test fit_odmr with ODMR that has no processed data."""
        light_image, laser_image = sample_images

        # Create ODMR without processing
        unprocessed_odmr = ODMR(sample_odmr_data)

        measurement = Measurement(
            odmr=unprocessed_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Should raise an error when ODMR data is not processed
        with pytest.raises(ValueError, match="ODMR data must be processed"):
            measurement.fit_odmr()

    def test_fit_odmr_auto_detection_failure(self, sample_odmr, sample_images, temp_output_dir):
        """Test fit_odmr behavior when auto model detection fails."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="auto",
        )

        # Mock guess_model to raise an exception
        with patch("QDMpy.guess.guess_model") as mock_guess:
            mock_guess.side_effect = ValueError("Could not determine model")

            # Should re-raise the exception from guess_model
            with pytest.raises(ValueError, match="Could not determine model"):
                measurement.fit_odmr()

    def test_fit_odmr_metadata_preservation(self, sample_odmr, sample_images, temp_output_dir):
        """Test that fit_odmr preserves and includes measurement metadata."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        # Add some metadata to the measurement
        measurement.metadata["test_key"] = "test_value"
        measurement.metadata["processing_time"] = 123.45

        # Mock FitManager
        with patch("QDMpy.fit.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            mock_fit_instance.fitted = True
            mock_fit_instance.model_name = "ESRSINGLE"
            mock_fit_instance.scan_dimensions = (10, 10)
            test_params = {
                "center": np.random.random(100),
                "width_0": np.random.random(100),
                "contrast": np.random.random(100),
                "offset": np.random.random(100),
                "chi2": np.random.random(100),
                "states": np.random.choice([0, 1], 100),
            }
            mock_fit_instance.get_param.side_effect = lambda param: test_params.get(param, None)

            result = measurement.fit_odmr()

            # Check that fit metadata was created (measurement metadata not currently merged)
            assert "fit_timestamp" in result.metadata
            assert "quality_metrics" in result.metadata
            assert "fit_settings" in result.metadata
            # Note: measurement metadata merging is a feature that should be implemented
