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
    from QDMpy.result import FitResult

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
        # Store default fit model preference
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

    def fit_odmr(
        self,
        model_name: str | None = None,
        **kwargs: Any
    ) -> 'FitResult':
        """Fit ODMR spectra and return results object.
        
        This method performs spectral fitting on the processed ODMR data using
        the specified model and returns a FitResult object containing the results
        and analysis methods.
        
        Args:
            model_name: Name of the model to use for fitting. Options are:
                - "ESR14N": For 14N isotope (3 peaks)
                - "ESR15N": For 15N isotope (2 peaks)  
                - "ESRSINGLE": For single resonance
                - None: Automatic model selection based on data analysis
            **kwargs: Additional arguments passed to FitManager (e.g., constraints, 
                estimator settings)
        
        Returns:
            FitResult object containing fit results and analysis methods
            
        Raises:
            ValueError: If ODMR data hasn't been processed yet
            ImportError: If required fitting dependencies are not available
        """
        # Import here to avoid circular imports
        from QDMpy.fit import FitManager
        from QDMpy.result import FitResult

        # Auto-detect model if none specified
        if model_name is None:
            LOG.info("Auto-detecting optimal model for ODMR data...")
            try:
                from QDMpy.guess import guess_model
                processed_data = self.odmr.processed_data
                # Use mean spectrum for model detection
                mean_spectrum = np.mean(processed_data.data, axis=(0, 1, 2))
                detected_model = guess_model(mean_spectrum)
                model_name = detected_model.name
                LOG.info("Auto-detected model: %s", model_name)
            except Exception as e:
                LOG.warning("Model auto-detection failed: %s. Using default.", e)
                model_name = self._fit_model

        LOG.info("Starting ODMR fitting with model: %s", model_name)

        # Validate that data has been processed
        try:
            processed_data = self.odmr.processed_data
            if processed_data is None:
                raise ValueError("ODMR data must be processed before fitting")
        except (AttributeError, ValueError):
            raise ValueError(
                "ODMR data must be processed before fitting. "
                "Call odmr.process_data() first."
            )

        # Check for fitting dependencies
        from QDMpy import PYGPUFIT_PRESENT
        if not PYGPUFIT_PRESENT:
            raise ImportError(
                "pyGpufit is required for fitting but not available. "
                "Please install pyGpufit to enable fitting functionality."
            )

        # Initialize FitManager with processed data
        LOG.debug("Initializing FitManager with data shape: %s", processed_data.data.shape)
        fit_manager = FitManager(
            data=processed_data.data,
            frequencies=processed_data.frequencies,
            model_name=model_name,
            **kwargs
        )

        # Perform the fitting
        LOG.info("Executing ODMR spectral fitting...")
        fit_manager.fit_odmr()

        # Extract fit results data (no heavy object references)
        LOG.debug("Extracting fit results data...")

        # Get all available parameters from FitManager
        parameters = {}
        for param_name in ['center', 'width_0', 'contrast', 'offset', 'chi2', 'states']:
            try:
                parameters[param_name] = fit_manager.get_param(param_name)
            except (KeyError, AttributeError):
                LOG.debug("Parameter '%s' not available for model %s", param_name, model_name)
                continue

        # Add any model-specific parameters
        if model_name == "ESR14N":
            # 14N has multiple width parameters
            for width_param in ['width_1', 'width_2']:
                try:
                    parameters[width_param] = fit_manager.get_param(width_param)
                except (KeyError, AttributeError):
                    continue
        elif model_name == "ESR15N":
            # 15N has width_1 parameter
            try:
                parameters['width_1'] = fit_manager.get_param('width_1')
            except (KeyError, AttributeError):
                pass

        # Calculate quality metrics
        quality_metrics = {}
        if 'chi2' in parameters and 'states' in parameters:
            chi2_values = parameters['chi2']
            states_values = parameters['states']
            quality_metrics = {
                'mean_chi2': float(np.mean(chi2_values)),
                'median_chi2': float(np.median(chi2_values)),
                'std_chi2': float(np.std(chi2_values)),
                'convergence_rate': float(np.mean(states_values == 0)),
                'n_pixels': int(chi2_values.size),
                'n_converged': int(np.sum(states_values == 0))
            }

        # Prepare metadata
        metadata = {
            'fit_timestamp': __import__('datetime').datetime.now().isoformat(),
            'quality_metrics': quality_metrics,
            'fit_settings': kwargs  # Store any additional fitting parameters
        }

        # Create lightweight FitResult with extracted data only
        result = FitResult(
            parameters=parameters,
            scan_dimensions=tuple(processed_data.scan_dimensions),
            pixel_spacing=self.pixel_spacing,
            model_name=model_name,
            metadata=metadata
        )

        LOG.info("ODMR fitting completed successfully")
        LOG.info("Extracted %d parameters for %d pixels",
                len(parameters), np.prod(processed_data.scan_dimensions))
        return result


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

    import matplotlib.pyplot as plt
    plt.imshow(measure.odmr.processed_data.data[0,0,:,0].reshape(measure.odmr.processed_data.scan_dimensions))
    plt.show()
