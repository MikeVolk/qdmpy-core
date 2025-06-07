"""Comprehensive measurement management for Quantum Diamond Microscopy.

This module provides the central `Measurement` class that serves as the primary interface
for working with Quantum Diamond Microscope (QDM) experiments. Key capabilities include:

- Data integration: Combines ODMR spectral data with optical images
- Spatial analysis: Maps spectral properties across the spatial dimensions
- Image processing: Handles light and laser reference images
- Metadata tracking: Maintains experiment parameters and processing history
- Output management: Organizes results in a structured directory hierarchy
- Statistical analysis: Identifies outliers and performs data quality assessment

The Measurement class integrates data from the ODMR module with optical images and
provides a unified interface for analysis and visualization of QDM experiments.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from os import PathLike

if not __package__:
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go one level up to the package root
    package_root = os.path.abspath(os.path.join(current_dir, '..'))
    # Add to path if not already there
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


# Following import must be after setup_package_paths
from QDMpy.odmr.odmr import ODMR

LOG = logging.getLogger(__name__)


class Measurement:
    """The Measurement class encapsulates all data and processing related to a single QDM
    (Quantum Diamond Microscope) measurement.

    It manages:
        - Raw and processed ODMR data using the ODMR instance.
        - Associated images (light and laser).
        - Fitting operations via external fitting instances.

    Attributes:
        odmr (ODMR): Instance managing ODMR data and processing.
        light_image (NDArray): Light image array with shape (height, width).
        laser_image (NDArray): Laser image array with shape (height, width).
        output_directory (Path): Path to the output directory.
        pixel_spacing (float): Spacing between pixels in meters.
        _outliers (Optional[NDArray]): Boolean mask for outlier pixels.
        _B111 (Optional[NDArray]): B111 field array, populated after fitting.
        _fit_model (str): Name of the model used for fitting ODMR spectra.
        metadata (Dict[str, Any]): Additional metadata for the measurement.
    """

    def __init__(
        self,
        odmr: ODMR,
        light_image: NDArray,
        laser_image: NDArray,
        output_directory: str | Path | PathLike,
        pixel_spacing: float = 4e-6,
        fit_model: str = 'auto',
    ) -> None:
        """Initialize the Measurement object.

        Args:
            odmr (ODMR): An initialized ODMR instance containing ODMR data.
            light_image (NDArray): Light image array with shape (height, width).
            laser_image (NDArray): Laser image array with shape (height, width).
            output_directory (Union[str, Path, PathLike]): Path to the output directory.
            pixel_spacing (float): Spacing between pixels in meters (pixel size).
                Default is 4 µm (4e-6).
            fit_model (str): Name of the model used for fitting ODMR spectra. Default is "auto".
                            If "auto", the model is chosen based on the mean ODMR data.

        Raises:
            ValueError: If the ODMR instance is not properly initialized or if image shapes
                       don't match the ODMR data.
        """
        LOG.info('Initializing Measurement object.')
        LOG.info('Output directory: "%s"', output_directory)

        self.output_directory = Path(output_directory)
        self.pixel_spacing = pixel_spacing
        self.metadata: dict[str, Any] = {}

        # Store the ODMR instance
        LOG.debug('Setting ODMR data.')
        self.odmr = odmr

        # Validate ODMR data availability
        try:
            # Use public property instead of accessing protected member
            _ = self.odmr.raw_data
        except ValueError:
            raise ValueError('ODMR instance has no raw data')

        # Validate ODMR instance data
        LOG.debug('ODMR raw data shape: %s', self.odmr.raw_data.shape)

        # Check if data has been processed
        try:
            LOG.debug('ODMR processed data shape: %s', self.odmr.processed_data.shape)
        except ValueError:
            LOG.warning('ODMR data has not been processed yet. Some functionality may be limited.')

        LOG.debug('ODMR frequencies shape: %s', self.odmr.raw_data.frequencies.shape)

        # Initialize outlier mask
        LOG.debug('Initializing outlier mask.')
        self._outliers: NDArray | None = np.ones(self.odmr.raw_data.shape, dtype=bool)

        # Store light and laser images
        LOG.debug('Storing light and laser images.')
        self.light_image = light_image
        self.laser_image = laser_image

        # Initialize B111 field and fit model
        LOG.debug('Initializing B111 field and fit model.')
        self._B111: NDArray | None = None
        # Placeholder for future fit integration
        self._fit_model = fit_model

    def __str__(self) -> str:
        """Return a string representation of the Measurement object.

        Returns:
            str: A human-readable string representation of the Measurement.
        """
        return (f"Measurement(odmr={self.odmr}, "
                f"output_directory='{self.output_directory}', "
                f"pixel_spacing={self.pixel_spacing} m)")

    def __repr__(self) -> str:
        """Return a developer string representation of the Measurement object.

        Returns:
            str: A detailed string representation for debugging and development.
        """
        return (f"Measurement(odmr={self.odmr!r}, "
                f"light_image.shape={self.light_image.shape}, "
                f"laser_image.shape={self.laser_image.shape}, "
                f"output_directory='{self.output_directory}', "
                f"pixel_spacing={self.pixel_spacing})")


if __name__ == '__main__':
    import numpy as np

    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor, FluorescenceCorrectionProcessor

    LOG.setLevel(logging.DEBUG)
    # User-friendly initialization with proper paths
    data_folder = "/home/mike/git/QDMpy/tests/data/FOV18x"
    loader = MatlabLoader(data_folder=data_folder)
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.processor_manager.add_processor(FluorescenceCorrectionProcessor())
    odmr.process_data()

    # Create dummy image data for testing
    dummy_light = np.ones((10, 10))
    dummy_laser = np.ones((10, 10))

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'output')
    os.makedirs(output_dir, exist_ok=True)

    measure = Measurement(
        odmr,
        dummy_light,
        dummy_laser,
        output_dir,
    )

    import matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(measure.odmr.processed_data.data[0,0,:,0].reshape(measure.odmr.processed_data.scan_dimensions))
    plt.show()
