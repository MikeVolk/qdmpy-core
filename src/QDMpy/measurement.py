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

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import Self

from QDMpy.odmr.odmr import ODMR

if TYPE_CHECKING:
    from os import PathLike

    from QDMpy.result import FitResult


class Measurement:
    """Encapsulate all data and processing for a single QDM measurement.

    The Measurement class encapsulates all data and processing related to a single QDM
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

    def __init__(  # noqa: PLR0913
        self: Self,
        odmr: ODMR,
        light_image: NDArray,
        laser_image: NDArray,
        output_directory: str | Path | PathLike,
        pixel_spacing: float = 4e-6,
        fit_model: str = "auto",
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
        logger.info("Initializing Measurement object.")
        logger.info(f'Output directory: "{output_directory}"')

        self.output_directory = Path(output_directory)
        self.pixel_spacing = pixel_spacing
        self.metadata: dict[str, Any] = {}

        # Store the ODMR instance
        logger.debug("Setting ODMR data.")
        self.odmr = odmr

        # Validate ODMR data availability
        try:
            # Use public property instead of accessing protected member
            _ = self.odmr.raw_data
        except ValueError as e:
            raise ValueError("ODMR instance has no raw data") from e

        # Validate ODMR instance data
        logger.debug(f"ODMR raw data shape: {self.odmr.raw_data.shape}")

        # Check if data has been processed
        try:
            logger.debug(f"ODMR processed data shape: {self.odmr.processed_data.shape}")
        except ValueError:
            logger.warning(
                "ODMR data has not been processed yet. Some functionality may be limited."
            )

        logger.debug(f"ODMR frequencies shape: {self.odmr.raw_data.frequencies.shape}")

        # Initialize outlier mask
        logger.debug("Initializing outlier mask.")
        self._outliers: NDArray | None = np.ones(self.odmr.raw_data.shape, dtype=bool)

        # Store light and laser images
        logger.debug("Storing light and laser images.")
        self.light_image = light_image
        self.laser_image = laser_image

        # Initialize B111 field and fit model
        logger.debug("Initializing B111 field and fit model.")
        self._B111: NDArray | None = None
        # Store default fit model preference
        self._fit_model = fit_model

    def __str__(self: Self) -> str:
        """Return a string representation of the Measurement object.

        Returns:
            str: A human-readable string representation of the Measurement.
        """
        return (
            f"Measurement(odmr={self.odmr}, "
            f"output_directory='{self.output_directory}', "
            f"pixel_spacing={self.pixel_spacing} m)"
        )

    def __repr__(self: Self) -> str:
        """Return a developer string representation of the Measurement object.

        Returns:
            str: A detailed string representation for debugging and development.
        """
        return (
            f"Measurement(odmr={self.odmr!r}, "
            f"light_image.shape={self.light_image.shape}, "
            f"laser_image.shape={self.laser_image.shape}, "
            f"output_directory='{self.output_directory}', "
            f"pixel_spacing={self.pixel_spacing})"
        )

    def _detect_model(self: Self, model_name: str | None) -> str:
        """Detect or validate the ODMR fitting model name.

        Args:
            model_name: Explicit model name, or None for auto-detection.

        Returns:
            Resolved model name string.
        """
        if model_name is not None:
            return model_name

        logger.info("Auto-detecting optimal model for ODMR data...")
        try:
            from QDMpy.guess import guess_model

            processed_data = self.odmr.processed_data
            values = processed_data.data.values
            n_pol, n_frange = values.shape[0], values.shape[1]
            n_freq = values.shape[-1]
            flat_data = values.reshape(n_pol, n_frange, -1, n_freq)
            detected = guess_model(flat_data)
            logger.info(f"Auto-detected model: {detected.name}")
            return detected.name
        except Exception as e:
            logger.warning(f"Model auto-detection failed: {e}. Using default.")
            return self._fit_model

    def _validate_fit_prerequisites(self: Self) -> Any:
        """Validate that processed data and GPU fitting are available.

        Returns:
            The ProcessedData object.

        Raises:
            ValueError: If ODMR data hasn't been processed.
            ImportError: If pyGpufit is not available.
        """
        try:
            processed_data = self.odmr.processed_data
            if processed_data is None:
                raise ValueError("ODMR data must be processed before fitting")
        except (AttributeError, ValueError) as e:
            raise ValueError(
                "ODMR data must be processed before fitting. "
                "Call odmr.process_data() first."
            ) from e

        from QDMpy import is_pygpufit_available

        if not is_pygpufit_available():
            raise ImportError(
                "pyGpufit is required for fitting but not available. "
                "Please install pyGpufit to enable fitting functionality."
            )
        return processed_data

    @staticmethod
    def _extract_fit_parameters(
        fit_manager: Any, model_name: str
    ) -> dict[str, NDArray]:
        """Extract fitted parameters and chi2 from a FitManager.

        Args:
            fit_manager: A fitted FitManager instance.
            model_name: Model name (for logging).

        Returns:
            Dictionary of parameter name to NDArray.
        """
        parameters: dict[str, NDArray] = {}
        for param_name in [*fit_manager.model_params_unique, 'chi2']:
            try:
                parameters[param_name] = fit_manager.get_param(param_name)
            except (KeyError, AttributeError, ValueError):
                logger.debug(f"Parameter '{param_name}' not available for model {model_name}")
        return parameters

    @staticmethod
    def _compute_quality_metrics(parameters: dict[str, NDArray]) -> dict[str, float]:
        """Compute fit quality metrics from extracted parameters.

        Args:
            parameters: Dictionary containing at least 'chi2' and optionally 'states'.

        Returns:
            Dictionary of quality metric names to values (empty if chi2/states missing).
        """
        if 'chi2' not in parameters or 'states' not in parameters:
            return {}
        chi2_values = parameters['chi2']
        states_values = parameters['states']
        return {
            'mean_chi2': float(np.mean(chi2_values)),
            'median_chi2': float(np.median(chi2_values)),
            'std_chi2': float(np.std(chi2_values)),
            'convergence_rate': float(np.mean(states_values == 0)),
            'n_pixels': int(chi2_values.size),
            'n_converged': int(np.sum(states_values == 0)),
        }

    def fit_odmr(
        self: Self,
        model_name: str | None = None,
        *,
        constraints: dict[str, Any] | None = None,
    ) -> FitResult:
        """Fit ODMR spectra and return results object.

        Args:
            model_name: Model name or None for auto-detection.
            constraints: Optional parameter constraints for fitting.

        Returns:
            FitResult object containing fit results and analysis methods.

        Raises:
            ValueError: If ODMR data hasn't been processed yet.
            ImportError: If required fitting dependencies are not available.
        """
        from QDMpy.fit import FitManager
        from QDMpy.result import FitResult

        model_name = self._detect_model(model_name)
        logger.info(f"Starting ODMR fitting with model: {model_name}")
        processed_data = self._validate_fit_prerequisites()

        fit_manager = FitManager(
            data=processed_data.data,
            frequencies=processed_data.frequencies,
            model_name=model_name,
            constraints=constraints,
        )
        fit_manager.fit_odmr()

        parameters = self._extract_fit_parameters(fit_manager, model_name)
        quality_metrics = self._compute_quality_metrics(parameters)

        import datetime

        metadata = {
            'fit_timestamp': datetime.datetime.now().isoformat(),
            'quality_metrics': quality_metrics,
            'fit_settings': {'constraints': constraints},
        }

        result = FitResult(
            parameters=parameters,
            scan_dimensions=tuple(processed_data.scan_dimensions),
            pixel_spacing=self.pixel_spacing,
            model_name=model_name,
            metadata=metadata,
        )

        logger.info("ODMR fitting completed successfully")
        logger.info(
            f"Extracted {len(parameters)} parameters for "
            f"{np.prod(processed_data.scan_dimensions)} pixels"
        )
        return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor, FluorescenceCorrectionProcessor

    logger.enable("QDMpy")
    data_folder = "/home/mike/git/QDMpy/tests/data/FOV18x"
    loader = MatlabLoader(data_folder=data_folder)
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.processor_manager.add_processor(FluorescenceCorrectionProcessor())
    odmr.process_data()

    dummy_light = np.ones((10, 10))
    dummy_laser = np.ones((10, 10))

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "output")
    os.makedirs(output_dir, exist_ok=True)

    measure = Measurement(
        odmr,
        dummy_light,
        dummy_laser,
        output_dir,
    )

    # xarray: select first polarity, first freq_range, first freq_idx -> 2D (y, x)
    plt.imshow(measure.odmr.processed_data.data.isel(polarity=0, freq_range=0, freq_idx=0).values)
    plt.show()
