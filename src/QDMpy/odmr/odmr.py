"""Core ODMR spectroscopy management for QDM analysis.

This module provides the central `ODMR` class which orchestrates all aspects of
Optically Detected Magnetic Resonance (ODMR) data handling. Key capabilities include:

- Data lifecycle: Managing raw and processed spectral data from acquisition to analysis
- Processing pipeline: Coordinating multiple sequential data transformations
- Frequency mapping: Tracking frequency values across different measurement ranges
- Polarization handling: Managing data from different microwave polarization states
- Spectral binning: Optimizing spectral resolution vs signal-to-noise ratio
- Outlier detection: Identifying and handling anomalous spectral data
- Fluorescence correction: Compensating for illumination variations across samples

The ODMR class integrates with ODMRData for storage and ODMRProcessorManager for
applying configurable processing pipelines to raw spectroscopic data.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.insert(0, project_root)

from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.processors import ODMRProcessorManager

LOG = logging.getLogger(__name__)


class ODMR:
    """Manages raw and processed ODMR data.

    The `ODMR` class provides functionality for loading raw data, applying a
    processing pipeline, and resetting data to its original state. It leverages
    the `ODMRProcessorManager` to handle processing tasks.

    Attributes:
        _raw_data (Optional[ODMRData]): The original raw data.
        _processed_data (Optional[ODMRData]): The processed data.
        processor_manager (ODMRProcessorManager): The processor manager for applying
                                                  transformations to the data.
    """

    def __init__(self, odmr_data: ODMRData | None = None) -> None:
        """Initialize the ODMR instance.

        Args:
            odmr_data (Optional[ODMRData]): An initial ODMRData instance to load.

        Attributes:
            _raw_data (Optional[ODMRData]): The original raw data.
            _processed_data (Optional[ODMRData]): The processed data.
            processor_manager (ODMRProcessorManager): The processor manager for applying
                                                     transformations to the data.
            is_processed (bool): Indicates whether the data has been processed.
        """
        self._raw_data = odmr_data
        self._processed_data = None
        self.is_processed = False  # Indicates whether the data has been processed
        self.processor_manager = ODMRProcessorManager()

    def load_data(
        self,
        raw_data: NDArray,
        scan_dimensions: NDArray,
        frequencies: NDArray,
    ) -> ODMR:
        """Load raw ODMR data into the instance.

        Args:
            raw_data (NDArray): Raw ODMR data as a 4D numpy array with shape:
                - Axis 0: Different polarities of measurements (typically 2 for positive/negative)
                - Axis 1: Different frequency ranges scanned in the experiment
                - Axis 2: Frequency points (number of frequency measurements per pixel)
                - Axis 3: Pixels (flattened from a 2D image with rows x cols pixels)
            scan_dimensions (NDArray): Image scan dimensions as (rows, cols).
                Used to reshape the flattened spatial pixels back to a 2D image.
            frequencies (NDArray): 1D array of frequencies used in the scan.

        Returns:
            ODMR: Self for method chaining.

        Sets:
            _raw_data: A new ODMRData instance with the provided data.
            _processed_data: Resets to None.
            is_processed: Resets to False.
        """
        LOG.info("Loading data into ODMR instance.")
        self._raw_data = ODMRData(raw_data, scan_dimensions, frequencies)
        self._processed_data = None
        self.is_processed = False
        return self

    def reset(self) -> ODMR:
        """Reset to the raw data.

        Discards any processing and reverts the instance to the original raw data.

        Returns:
            ODMR: Self for method chaining.

        Raises:
            ValueError: If no raw data is loaded.
        """
        if self._raw_data is None:
            LOG.error("No raw data loaded. Cannot reset.")
            raise ValueError("No raw data to reset to.")
        LOG.info("Resetting to raw data.")
        self._processed_data = None
        self.is_processed = False
        return self

    def process_data(self) -> ODMR:
        """Apply the processing pipeline to the raw data.

        Uses the `ODMRProcessorManager` to process the raw data and stores the result
        in `_processed_data`.

        Returns:
            ODMR: Self for method chaining.

        Raises:
            ValueError: If no raw data is loaded.
        """
        if self._raw_data is None:
            LOG.error("No data loaded.")
            raise ValueError("No ODMRData loaded.")
        LOG.info("Processing data.")
        self._processed_data = self.processor_manager.process(self._raw_data)
        self.is_processed = True
        return self

    @property
    def raw_data(self) -> ODMRData:
        """Access the raw ODMRData.

        Returns:
            ODMRData: The original raw data instance.

        Raises:
            ValueError: If no raw data is loaded.
        """
        if self._raw_data is None:
            LOG.error("No raw data loaded.")
            raise ValueError("No raw data available.")
        return self._raw_data

    @property
    def processed_data(self) -> ODMRData:
        """Access the processed ODMRData.

        Returns:
            ODMRData: The processed data instance.

        Raises:
            ValueError: If no processing has been performed yet.
        """
        if self._processed_data is None:
            LOG.error("No data has been processed yet.")
            raise ValueError("No processed data available.")
        return self._processed_data


# Example usage (uncomment to run)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor

    # User-friendly initialization
    loader = MatlabLoader(data_folder="/home/mike/git/QDMpy/tests/data/FOV18x")
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.process_data()

    # Access data
    print(odmr.raw_data.shape)
    print(odmr.processed_data.shape)

    plt.plot(odmr.processed_data.frequencies, odmr.processed_data.data[0, 0, 0, :])
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Signal (a.u.)")
    plt.title("Processed ODMR Signal")
    plt.show()
