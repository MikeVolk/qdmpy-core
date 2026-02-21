"""Core ODMR spectroscopy management for QDM analysis.

This module provides the central `ODMR` class which orchestrates all aspects of
ODMR data handling including raw/processed data lifecycle and processing pipelines.
"""

from __future__ import annotations

from typing import Self

import xarray as xr
from loguru import logger
from numpy.typing import NDArray

from QDMpy.exceptions import DataNotLoadedError
from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.processors import ODMRProcessorManager


class ODMR:
    """Manages raw and processed ODMR data.

    Attributes:
        _raw_data: The original raw data.
        _processed_data: The processed data.
        processor_manager: The processor manager for applying transformations.
    """

    def __init__(self: Self, odmr_data: ODMRData | None = None) -> None:
        """Initialize the ODMR manager with optional raw data.

        Args:
            odmr_data: Optional ODMRData instance to initialize with.
        """
        self._raw_data = odmr_data
        self._processed_data: ODMRData | None = None
        self.is_processed = False
        self.processor_manager = ODMRProcessorManager()

    def load_data(
        self: Self,
        raw_data: NDArray,
        scan_dimensions: tuple[int, int],
        frequencies: NDArray,
    ) -> ODMR:
        """Load raw ODMR data into the instance.

        Args:
            raw_data: 4D numpy array (n_pol, n_frange, n_pixels, n_freqs).
            scan_dimensions: Image scan dimensions as (rows, cols).
            frequencies: Frequency array in Hz.

        Returns:
            Self for method chaining.
        """
        logger.info("Loading data into ODMR instance.")
        self._raw_data = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies)
        self._processed_data = None
        self.is_processed = False
        return self

    def load_xarray(self: Self, data: xr.DataArray) -> ODMR:
        """Load an xr.DataArray directly.

        Args:
            data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).

        Returns:
            Self for method chaining.
        """
        logger.info("Loading xarray data into ODMR instance.")
        self._raw_data = ODMRData(data)
        self._processed_data = None
        self.is_processed = False
        return self

    def reset(self: Self) -> ODMR:
        """Reset to the raw data."""
        if self._raw_data is None:
            logger.error("No raw data loaded. Cannot reset.")
            msg = "No raw data to reset to."
            raise DataNotLoadedError(msg)
        logger.info("Resetting to raw data.")
        self._processed_data = None
        self.is_processed = False
        return self

    def process_data(self: Self) -> ODMR:
        """Apply the processing pipeline to the raw data."""
        if self._raw_data is None:
            logger.error("No data loaded.")
            msg = "No ODMRData loaded."
            raise DataNotLoadedError(msg)
        logger.info("Processing data.")
        self._processed_data = self.processor_manager.process(self._raw_data)
        self.is_processed = True
        return self

    @property
    def raw_data(self: Self) -> ODMRData:
        """Access the raw ODMRData."""
        if self._raw_data is None:
            logger.error("No raw data loaded.")
            msg = "No raw data available."
            raise DataNotLoadedError(msg)
        return self._raw_data

    @property
    def processed_data(self: Self) -> ODMRData:
        """Access the processed ODMRData."""
        if self._processed_data is None:
            logger.error("No data has been processed yet.")
            msg = "No processed data available."
            raise DataNotLoadedError(msg)
        return self._processed_data
