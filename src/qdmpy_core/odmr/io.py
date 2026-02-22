"""Data loading interface for ODMR spectroscopy data.

This module provides a modular framework for loading Optically Detected Magnetic
Resonance (ODMR) data from various file formats through a collection of loader classes.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Self

import mat73
import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from scipy.io import loadmat

from qdmpy_core.exceptions import DataLoadError
from qdmpy_core.odmr.data import FRANGE_LABELS, POLARITY_LABELS


class BaseLoader(ABC):
    """Abstract base class for ODMR data loaders.

    Subclasses should implement the `load` method to provide specific functionality
    for loading ODMR data from different file types or sources.

    The loaded data should be an xr.DataArray with named dimensions:
    - polarity: measurement field polarity (pos/neg)
    - freq_range: low vs high frequency band
    - y: spatial row
    - x: spatial column
    - freq_idx: frequency sweep index within a range
    """

    @abstractmethod
    def load(self: Self) -> xr.DataArray:
        """Load ODMR data.

        Returns:
            xr.DataArray with dims (polarity, freq_range, y, x, freq_idx)
            and a 'freq_ghz' coordinate giving actual GHz values per
            (freq_range, freq_idx).
        """


class MatlabLoader(BaseLoader):
    """Loader for ODMR data from MATLAB files.

    Attributes:
        data_folder: Path to the folder containing MATLAB files.
    """

    def __init__(self: Self, data_folder: str) -> None:
        """Initialize the MATLAB loader with a data folder path.

        Args:
            data_folder: Path to the folder containing MATLAB data files.
        """
        self.data_folder = data_folder

    def load(self: Self) -> xr.DataArray:  # noqa: C901, PLR0912, PLR0915
        """Load ODMR data from the specified folder.

        Returns:
            xr.DataArray with dims (polarity, freq_range, y, x, freq_idx)

        Raises:
            FileNotFoundError: If no valid MATLAB files are found.
            ValueError: If the MATLAB file contains an unsupported structure.
        """
        files = sorted(
            f for f in os.listdir(self.data_folder) if f.endswith(".mat") and "run_" in f
        )
        if not files:
            msg = "No valid MATLAB files found in the folder."
            raise DataLoadError(msg)

        logger.info(f"Found {len(files)} MATLAB file(s) in {self.data_folder}")
        t_start = time.perf_counter()

        per_file_data: list[NDArray] = []
        rows: int = 0
        cols: int = 0
        frequencies: NDArray | None = None

        for file in files:
            full_path = os.path.join(self.data_folder, file)
            logger.debug(f"Loading MATLAB file: {file}")
            try:
                mat_data = mat73.loadmat(full_path)
                logger.debug(f"Loaded {file} with mat73")
            except Exception:
                mat_data = loadmat(full_path)
                logger.debug(f"Loaded {file} with scipy.io.loadmat (mat73 fallback)")

            stacked_data = self._process_mat_file(mat_data)
            logger.debug(f"Extracted data shape: {stacked_data.shape} from {file}")
            per_file_data.append(stacked_data)

            try:
                rows = int(np.squeeze(mat_data["imgNumRows"]))
                cols = int(np.squeeze(mat_data["imgNumCols"]))
            except KeyError as e:
                msg = f"Missing required key in MATLAB file: {e}"
                raise DataLoadError(msg) from e

            try:
                freq_list = np.squeeze(mat_data["freqList"])
                if "numFreqs" in mat_data:
                    n_freqs = int(np.squeeze(mat_data["numFreqs"]))
                    if n_freqs != len(freq_list):
                        frequencies = np.array([freq_list[:n_freqs], freq_list[n_freqs:]])
                    else:
                        frequencies = freq_list
                else:
                    frequencies = freq_list
            except KeyError as e:
                msg = f"Missing required key in MATLAB file: {e}"
                raise DataLoadError(msg) from e

        if frequencies is None:
            msg = "No frequency data found in MATLAB files."
            raise DataLoadError(msg)

        # Stack polarity axis from multiple files
        if len(per_file_data) == 1:
            raw_data = per_file_data[0][np.newaxis, ...]
        else:
            raw_data = np.stack(per_file_data, axis=0)

        # raw_data shape: (n_pol, n_frange, n_pixels, n_freqs)
        n_pol, n_frange, _n_pixels, n_freqs = raw_data.shape

        # Reshape flattened pixels to 2D spatial grid
        raw_data = raw_data.reshape(n_pol, n_frange, rows, cols, n_freqs)
        logger.debug(f"Reshaped data to ({n_pol}, {n_frange}, {rows}, {cols}, {n_freqs})")

        # Build frequency coordinate
        if frequencies.ndim == 1:
            freq_ghz = np.tile(frequencies, (n_frange, 1)) / 1e9
        else:
            freq_ghz = frequencies / 1e9

        logger.debug(f"Frequency range: {freq_ghz.min():.4f} - {freq_ghz.max():.4f} GHz")

        elapsed = time.perf_counter() - t_start
        logger.info(f"MATLAB data loaded in {elapsed:.2f}s — shape {raw_data.shape}")

        polarity_labels = POLARITY_LABELS[:n_pol]
        frange_labels = FRANGE_LABELS[:n_frange]

        return xr.DataArray(
            raw_data,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": polarity_labels,
                "freq_range": frange_labels,
                "freq_ghz": (["freq_range", "freq_idx"], freq_ghz),
            },
        )

    @staticmethod
    def _process_mat_file(mat_file: dict[str, Any]) -> NDArray:
        """Process a MATLAB file to extract raw ODMR data.

        Returns:
            NDArray with shape (n_frange, n_pixels, n_freqs).

        Raises:
            ValueError: If the MATLAB file contains an unsupported number of image stacks.
        """
        DUAL_POLARITY_STACKS = 2  # noqa: N806
        QUAD_POLARITY_STACKS = 4  # noqa: N806
        n_img_stacks = len([k for k in mat_file if "imgStack" in k])
        logger.debug(f"Found {n_img_stacks} image stacks in MATLAB file")
        if n_img_stacks == DUAL_POLARITY_STACKS:
            return np.stack([mat_file["imgStack1"].T, mat_file["imgStack2"].T], axis=0)
        if n_img_stacks == QUAD_POLARITY_STACKS:
            stack_low = np.concatenate([mat_file["imgStack1"], mat_file["imgStack2"]], axis=0).T
            stack_high = np.concatenate([mat_file["imgStack3"], mat_file["imgStack4"]], axis=0).T
            return np.stack([stack_low, stack_high], axis=0)
        msg = "Unsupported number of image stacks in MATLAB file."
        raise DataLoadError(msg)
