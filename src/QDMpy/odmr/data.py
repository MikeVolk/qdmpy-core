"""Data structures for ODMR spectroscopy in QDM analysis.

Convention: All frequency values are in GHz. Raw Hz input from MATLAB files
is converted at the ``from_numpy()`` input boundary.

This module provides the `ODMRData` class, which serves as the fundamental data
structure for handling Optically Detected Magnetic Resonance (ODMR) spectroscopic
measurements using xarray for named-dimension data representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import Self

if TYPE_CHECKING:
    from QDMpy.odmr.io import BaseLoader


class ODMRData:
    """Represents raw and processed ODMR data backed by an xr.DataArray.

    The underlying DataArray has five named dimensions:
    - polarity: measurement field polarity
    - freq_range: low vs high frequency band
    - y: spatial row
    - x: spatial column
    - freq_idx: frequency sweep index within a range

    Attributes:
        data: xr.DataArray with the named dimensions above.
        metadata: Additional metadata associated with the data.
    """

    def __init__(
        self: Self,
        data: xr.DataArray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.metadata = metadata or {}

    @classmethod
    def from_loader(
        cls,
        loader: BaseLoader,
        loader_args: dict[str, Any] | None = None,
    ) -> ODMRData:
        """Create an ODMRData instance using a loader.

        Args:
            loader: An instantiated loader to fetch data dynamically.
            loader_args: Arguments to pass to the loader (optional).

        Returns:
            An instance populated with data loaded from the loader.

        Raises:
            RuntimeError: If the loader fails to fetch data.
        """
        logger.info(f'Loading ODMR data using loader: {loader.__class__.__name__}')
        try:
            data = loader.load(**(loader_args or {}))
            return cls(data)
        except Exception as e:
            logger.exception(
                f'Failed to load data using loader {loader.__class__.__name__}: {e}'
            )
            raise RuntimeError(f'Data loading failed: {e}') from e

    @classmethod
    def from_numpy(
        cls,
        data: NDArray,
        scan_dimensions: tuple[int, int],
        frequencies: NDArray,
        metadata: dict[str, Any] | None = None,
    ) -> ODMRData:
        """Create ODMRData from a raw numpy array (convenience constructor).

        Args:
            data: 4D numpy array with shape (n_pol, n_frange, n_pixels, n_freqs).
            scan_dimensions: Spatial dimensions as (rows, cols).
            frequencies: Frequency array in Hz. Shape (n_freqs,) or
                (n_frange, n_freqs).
            metadata: Optional metadata dict.

        Returns:
            ODMRData instance wrapping the data as an xr.DataArray.
        """
        n_pol, n_frange = data.shape[0], data.shape[1]
        n_freqs = data.shape[-1]
        rows, cols = scan_dimensions

        data_5d = data.reshape(n_pol, n_frange, rows, cols, n_freqs)

        frequencies = np.asarray(frequencies)
        if frequencies.ndim == 1:
            freq_ghz = np.tile(frequencies, (n_frange, 1)) / 1e9
        else:
            freq_ghz = frequencies / 1e9

        polarity_labels = [f'pol_{i}' for i in range(n_pol)]
        frange_labels = [f'frange_{i}' for i in range(n_frange)]

        da = xr.DataArray(
            data_5d,
            dims=('polarity', 'freq_range', 'y', 'x', 'freq_idx'),
            coords={
                'polarity': polarity_labels,
                'freq_range': frange_labels,
                'freq_ghz': (['freq_range', 'freq_idx'], freq_ghz),
            },
        )
        return cls(da, metadata=metadata)

    @property
    def scan_dimensions(self: Self) -> tuple[int, int]:
        """Spatial dimensions (rows, cols) derived from the DataArray."""
        return (self.data.sizes['y'], self.data.sizes['x'])

    @property
    def frequencies(self: Self) -> NDArray:
        """Frequency values in GHz as a numpy array.

        Returns shape (n_frange, n_freqs) or (n_freqs,) depending on the data.
        """
        return self.data.coords['freq_ghz'].values

    @property
    def numpy(self: Self) -> NDArray:
        """Raw numpy array for performance-critical code."""
        return self.data.values

    @property
    def shape(self: Self) -> tuple[int, ...]:
        """Shape of the underlying DataArray."""
        return self.data.shape
