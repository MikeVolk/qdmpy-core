"""Pytest validation tests for data loading functionality.

These tests validate that the new QDMpy codebase loads identical data
compared to reference data generated from the old codebase.
"""

from __future__ import annotations

import numpy as np
import pytest
from loguru import logger


@pytest.mark.validation
@pytest.mark.data_loading
class TestDataLoadingValidation:
    """Validation tests for ODMR data loading operations."""

    def test_raw_data_loading(self, reference_data, new_qdmpy_modules, test_data_folder) -> None:
        """Test that raw ODMR data is loaded identically to reference data."""
        # Import new codebase modules
        _QDMpy_new, _Measurement_new, _ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules

        # Load data with new codebase
        loader = MatlabLoader(data_folder=str(test_data_folder))
        odmr_data = ODMRData.from_loader(loader=loader)

        # Get reference data
        ref_raw_data = reference_data["raw_data"]
        ref_frequencies = reference_data["frequencies"]
        ref_scan_dims = reference_data["scan_dimensions"]

        # Compare raw data
        np.testing.assert_array_equal(
            odmr_data.data, ref_raw_data, err_msg="Raw ODMR data does not match reference"
        )

        # Compare frequencies (allow for float32/float64 differences)
        np.testing.assert_allclose(
            odmr_data.frequencies,
            ref_frequencies,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Frequencies do not match reference within tolerance",
        )

        # Compare scan dimensions
        np.testing.assert_array_equal(
            odmr_data.scan_dimensions,
            ref_scan_dims,
            err_msg="Scan dimensions do not match reference",
        )

        logger.info("Raw data loading validation passed")
        logger.info(f"   Data shape: {odmr_data.data.shape}")
        logger.info(f"   Scan dimensions: {odmr_data.scan_dimensions}")

    def test_reference_images_loading(self, test_data_folder) -> None:
        """Test that reference images (LED, laser) are loaded correctly."""
        led_path = test_data_folder / "LED.csv"
        laser_path = test_data_folder / "laser.csv"

        # Verify files exist
        assert led_path.exists(), f"LED.csv not found at {led_path}"
        assert laser_path.exists(), f"laser.csv not found at {laser_path}"

        # Load images
        led_image = np.genfromtxt(led_path, delimiter=",")
        laser_image = np.genfromtxt(laser_path, delimiter=",")

        # Basic validation
        assert led_image.size > 0, "LED image is empty"
        assert laser_image.size > 0, "Laser image is empty"
        assert led_image.ndim == 2, "LED image should be 2D"
        assert laser_image.ndim == 2, "Laser image should be 2D"

        logger.info("Reference images loading passed")
        logger.info(f"   LED shape: {led_image.shape}")
        logger.info(f"   Laser shape: {laser_image.shape}")
