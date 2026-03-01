"""Core ODMR spectroscopy management for QDM analysis.

This module provides the central `ODMR` class which orchestrates all aspects of
ODMR data handling including raw/processed data lifecycle and processing pipelines.
"""

from __future__ import annotations

from typing import Self

import xarray as xr
from loguru import logger
from numpy.typing import NDArray

from qdmpy.exceptions import DataNotLoadedError
from qdmpy.odmr.data import ODMRData
from qdmpy.odmr.processors import ODMRProcessorManager


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
        self._raw_data = ODMRData(data=data)
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

    def spectrum(
        self: Self,
        y: int,
        x: int,
        polarity: str = "neg",
        freq_range: str = "low",
        *,
        processed: bool = True,
    ) -> tuple[NDArray, NDArray]:
        """Return the ODMR spectrum at pixel (y, x) for one polarity and freq range.

        Args:
            y: Row index in the scan grid.
            x: Column index in the scan grid.
            polarity: Polarity label — 'neg' or 'pos'.
            freq_range: Frequency range label — 'low' or 'high'.
            processed: If True use processed_data, else raw_data.

        Returns:
            Tuple (freq_ghz, intensity) each shape (n_freq,).
        """
        data = self.processed_data if processed else self.raw_data
        freq = data.data.coords["freq_ghz"].sel(freq_range=freq_range).values
        intensity = data.data.sel(polarity=polarity, freq_range=freq_range).values[y, x, :]
        return freq, intensity

    def plot_spectra(
        self: Self,
        y: int,
        x: int,
        *,
        processed: bool = True,
    ) -> None:
        """Plot all ODMR spectra for pixel (y, x) in a polarity x freq_range grid.

        Each subplot shows one (polarity, freq_range) combination. Only the
        combinations present in the data are plotted.

        Args:
            y: Row index in the scan grid.
            x: Column index in the scan grid.
            processed: If True use processed_data, else raw_data.
        """
        from qdmpy.plotting import plot_odmr_spectra

        odmr_data = self.processed_data if processed else self.raw_data
        plot_odmr_spectra(odmr_data, y, x)
