"""
Module: QDMpy.odmr.odmr
=======================

This module defines the `ODMR` class, which orchestrates the management of raw and
processed ODMR (Optically Detected Magnetic Resonance) data. It utilizes the
`ODMRData` class for storing data and the `ODMRProcessorManager` for applying
processing pipelines.

Classes:
    - ODMR: Manages raw and processed ODMR data, including data loading, processing,
            and resetting.

Imports:
    - Python standard library: logging, sys, os
    - Local: QDMpy.odmr.data (ODMRData), QDMpy.odmr.processors
"""

from __future__ import annotations
import sys
import os

from typing import Optional, TYPE_CHECKING
import logging

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
    """
    Manages raw and processed ODMR data.

    The `ODMR` class provides functionality for loading raw data, applying a
    processing pipeline, and resetting data to its original state. It leverages
    the `ODMRProcessorManager` to handle processing tasks.

    Attributes:
        _raw_data (Optional[ODMRData]): The original raw data.
        _processed_data (Optional[ODMRData]): The processed data.
        processor_manager (ODMRProcessorManager): The processor manager for applying
                                                  transformations to the data.
    """

    def __init__(self, odmr_data: Optional[ODMRData] = None) -> None:
        """
        Initialize the ODMR instance.

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
        self, raw_data: NDArray, scan_dimensions: NDArray, frequencies: NDArray
    ) -> "ODMR":
        """
        Load raw ODMR data into the instance.

        Args:
            raw_data (NDArray): Raw ODMR data as a numpy array.
            scan_dimensions (NDArray): Image scan dimensions (rows, cols).
            frequencies (NDArray): Frequency list for the scan.

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

    def reset(self) -> "ODMR":
        """
        Reset to the raw data.

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

    def process_data(self) -> "ODMR":
        """
        Apply the processing pipeline to the raw data.

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
        """
        Access the raw ODMRData.

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
        """
        Access the processed ODMRData.

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
# if __name__ == "__main__":
#     from QDMpy.odmr.data import ODMRData
#     from QDMpy.odmr.io import MatlabLoader
#     from QDMpy.odmr.processors import BinningProcessor
# 
#     # User-friendly initialization
#     loader = MatlabLoader(data_folder="/path/to/data")
#     odmr_data = ODMRData.from_loader(loader=loader)
#     odmr = ODMR(odmr_data)
#     odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
#     odmr.process_data()
#     
#     # Access data
#     print(odmr.raw_data.shape)
#     print(odmr.processed_data.shape)
