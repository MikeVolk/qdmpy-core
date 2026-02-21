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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import matplotlib.image as mpimg
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from QDMpy.exceptions import DataLoadError, DataNotLoadedError, DependencyError
from QDMpy.odmr.manager import ODMR

if TYPE_CHECKING:
    from os import PathLike

    from QDMpy.fitting.result import FitResult
    from QDMpy.odmr.data import ODMRData


def has_csv(lst: Sequence[str | bytes | os.PathLike[Any]]) -> bool:
    """Check if a list of files contains a CSV file.

    Args:
        lst: List of file paths to check.

    Returns:
        True if at least one file has a .csv extension, False otherwise.
    """
    return any(".csv" in str(s).lower() for s in lst)


def get_image_file(lst: Sequence[str | bytes | os.PathLike[Any]]) -> str:
    """Get the path to the first image file in the list.

    Prefers CSV files if available, otherwise looks for JPG files.

    Args:
        lst: List of file paths to search.

    Returns:
        Path to the first suitable image file.

    Raises:
        DataLoadError: If no suitable image files are found.
    """
    if has_csv(lst):
        filtered_lst = [s for s in lst if ".csv" in str(s).lower()]
    else:
        filtered_lst = [s for s in lst if ".jpg" in str(s).lower()]

    if not filtered_lst:
        msg = "No suitable image files found in the list"
        raise DataLoadError(msg)

    selected = str(filtered_lst[0])
    logger.debug(f"Selected image file: {selected}")
    return selected


def get_image(
    folder: str | bytes | os.PathLike[Any],
    lst: Sequence[str | bytes | os.PathLike[Any]],
) -> NDArray:
    """Load an image from a file in the specified folder.

    Attempts to load a CSV file first, falling back to JPG if no CSV is available.

    Args:
        folder: Path to the folder containing the image files.
        lst: List of file names to search for image files.

    Returns:
        Image data as a numpy array.

    Raises:
        DataLoadError: If no suitable image files are found or if the image cannot be loaded.
    """
    folder_str = str(folder)

    try:
        image_file = get_image_file(lst)
        file_path = os.path.join(folder_str, image_file)
        logger.debug(f"Loading image from: {file_path}")

        if image_file.lower().endswith(".csv"):
            try:
                img = np.loadtxt(file_path, delimiter=",")
            except ValueError:
                logger.debug("CSV comma delimiter failed, falling back to whitespace")
                img = np.loadtxt(file_path)
        else:
            img = mpimg.imread(file_path)
    except Exception as e:
        msg = f"Failed to load image: {e!s}"
        raise DataLoadError(msg) from e
    else:
        result = np.array(img)
        logger.info(f"Loaded image {file_path} with shape {result.shape}")
        return result


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
        except (ValueError, DataNotLoadedError) as e:
            msg = "ODMR instance has no raw data"
            raise DataNotLoadedError(msg) from e

        # Validate ODMR instance data
        logger.debug(f"ODMR raw data shape: {self.odmr.raw_data.shape}")

        # Check if data has been processed
        try:
            logger.debug(f"ODMR processed data shape: {self.odmr.processed_data.shape}")
        except (ValueError, DataNotLoadedError):
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
            from QDMpy.fitting.guess import guess_model

            processed_data = self.odmr.processed_data
            values = processed_data.data.values
            n_pol, n_frange = values.shape[0], values.shape[1]
            n_freq = values.shape[-1]
            flat_data = values.reshape(n_pol, n_frange, -1, n_freq)
            detected = guess_model(flat_data)
        except Exception as e:
            logger.warning(f"Model auto-detection failed: {e}. Using default.")
            return self._fit_model
        else:
            logger.info(f"Auto-detected model: {detected.name}")
            return detected.name

    def _validate_fit_prerequisites(self: Self) -> ODMRData:
        """Validate that processed data and GPU fitting are available.

        Returns:
            The ProcessedData object.

        Raises:
            DataNotLoadedError: If ODMR data hasn't been processed.
            DependencyError: If pyGpufit is not available.
        """
        try:
            processed_data = self.odmr.processed_data
        except (AttributeError, ValueError, DataNotLoadedError) as e:
            msg = "ODMR data must be processed before fitting. Call odmr.process_data() first."
            raise DataNotLoadedError(msg) from e

        from QDMpy.settings import is_pygpufit_available

        if not is_pygpufit_available():
            msg = (
                "pyGpufit is required for fitting but not available. "
                "Please install pyGpufit to enable fitting functionality."
            )
            raise DependencyError(msg)
        return processed_data

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
            DataNotLoadedError: If ODMR data hasn't been processed yet.
            DependencyError: If required fitting dependencies are not available.
        """
        from QDMpy.fitting.manager import FitManager

        model_name = self._detect_model(model_name)
        logger.info(f"Starting ODMR fitting with model: {model_name}")
        processed_data = self._validate_fit_prerequisites()

        fit_manager = FitManager(model_name=model_name, constraints=constraints)
        result = fit_manager.fit(
            processed_data.data,
            processed_data.frequencies,
            pixel_spacing=self.pixel_spacing,
        )

        logger.info("ODMR fitting completed successfully")
        return result
