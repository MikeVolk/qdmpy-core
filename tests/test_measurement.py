"""
Test module for QDMpy.measurement

These tests cover the Measurement class, which encapsulates all data and processing
related to a single QDM (Quantum Diamond Microscope) measurement.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now we can import from QDMpy
from QDMpy.measurement import Measurement
from QDMpy.odmr.odmr import ODMR
from QDMpy.odmr.data import ODMRData
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
            fit_model="auto"
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
            output_directory=temp_output_dir
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
                output_directory=temp_output_dir
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
            output_directory=temp_output_dir
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
            pixel_spacing=pixel_spacing
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
            output_directory=output_dir_str
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
            output_directory=temp_output_dir
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
        
    @patch('logging.Logger.debug')
    def test_logging(self, mock_debug, sample_odmr, sample_images, temp_output_dir):
        """Test that initialization logs appropriate messages."""
        light_image, laser_image = sample_images
        
        # Create a measurement object
        Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir
        )
        
        # Check that the appropriate log messages were generated
        assert any("Setting ODMR data" in call.args[0] for call in mock_debug.call_args_list)
        assert any("Initializing outlier mask" in call.args[0] for call in mock_debug.call_args_list)
        assert any("Storing light and laser images" in call.args[0] for call in mock_debug.call_args_list)