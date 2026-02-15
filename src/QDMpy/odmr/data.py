"""Data structures for ODMR spectroscopy in QDM analysis.

This module provides the `ODMRData` class, which serves as the fundamental data structure
for handling Optically Detected Magnetic Resonance (ODMR) spectroscopic measurements.
Key capabilities include:

- Dimensional organization: Managing multi-dimensional data (spatial, spectral, polarization)
- Data representation: Consistent access to raw and processed spectral measurements
- Frequency mapping: Maintaining correspondence between data points and frequencies
- Metadata handling: Tracking experimental parameters and processing history
- Spatial information: Preserving the 2D spatial context of measurements
- Initialization options: Creating instances from raw arrays or through loaders
- Data validation: Ensuring data consistency across all dimensions

This class provides a unified representation of ODMR data throughout the analysis
pipeline, from raw measurements to processed spectra ready for fitting and analysis.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

from loguru import logger
from numpy.typing import NDArray

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.insert(0, project_root)


if TYPE_CHECKING:
    from QDMpy.odmr.io import BaseLoader


class ODMRData:
    """Represents raw and processed ODMR (Optically Detected Magnetic Resonance) data.

    Attributes:
        data (NDArray): The raw ODMR data, as a 4D numpy array with shape:
            - Axis 0: Different polarities of measurements (typically 2 for positive/negative)
            - Axis 1: Different frequency ranges scanned in the experiment
            - Axis 2: Spatial pixels (flattened from a 2D image with rows x cols pixels)
            - Axis 3: Frequency points (number of frequency measurements per pixel)
        scan_dimensions (NDArray): The dimensions of the scan as (rows, cols).
            Used to reshape the flattened spatial pixels back to a 2D image.
        frequencies (NDArray): A 1D array of frequencies used in the scan.
        metadata (dict): Additional metadata associated with the data.
    """

    def __init__(
        self,
        data: NDArray,
        scan_dimensions: NDArray,
        frequencies: NDArray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the ODMRData object.

        Args:
            data (NDArray): Raw ODMR data as a 4D numpy array with shape:
                - Axis 0: Different polarities of measurements (typically 2 for positive/negative)
                - Axis 1: Different frequency ranges scanned in the experiment
                - Axis 2: Spatial pixels (flattened from a 2D image with rows x cols pixels)
                - Axis 3: Frequency points (number of frequency measurements per pixel)
            scan_dimensions (NDArray): Image scan dimensions as (rows, cols).
                Used to reshape the flattened spatial pixels back to a 2D image.
            frequencies (NDArray): 1D array of frequencies used in the scan.
            metadata (Optional[Dict[str, Any]]): Additional metadata (optional).
        """
        self.data = data
        self.scan_dimensions = scan_dimensions
        self.frequencies = frequencies
        self.metadata = metadata or {}

    @classmethod
    def from_loader(
        cls,
        loader: BaseLoader,
        loader_args: dict[str, Any] | None = None,
    ) -> ODMRData:
        """Create an ODMRData instance using a loader.

        Args:
            loader (BaseLoader): An instantiated loader to fetch data dynamically.
            loader_args (Optional[Dict[str, Any]]): Arguments to pass to the loader
                                                    (optional).

        Returns:
            ODMRData: An instance populated with data loaded from the loader.

        Raises:
            RuntimeError: If the loader fails to fetch data.
        """
        logger.info(f"Loading ODMR data using loader: {loader.__class__.__name__}")
        try:
            raw_data, scan_dimensions, frequencies = loader.load(**(loader_args or {}))
            return cls(raw_data, scan_dimensions, frequencies)
        except Exception as e:
            logger.exception(
                f"Failed to load data using loader {loader.__class__.__name__}: {e}"
            )
            raise RuntimeError(f"Data loading failed: {e}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the raw ODMR data.

        Returns:
            tuple[int, ...]: The shape of the raw data as a tuple with format:
                (polarities, frequency_ranges, spatial_pixels, frequencies)

                For example: (2, 1, 10000, 501) would represent:
                - 2 polarities (positive/negative)
                - 1 frequency range
                - 10000 spatial pixels (e.g., 100x100 image flattened)
                - 501 frequency points per pixel
        """
        return self.data.shape
