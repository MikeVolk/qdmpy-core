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

from qdmpy.exceptions import DataLoadError
from qdmpy.odmr.data import FRANGE_LABELS, POLARITY_LABELS


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

    def load(self: Self) -> xr.DataArray:
        """Load ODMR data from the specified folder.

        Returns:
            xr.DataArray with dims (polarity, freq_range, y, x, freq_idx)

        Raises:
            FileNotFoundError: If no valid MATLAB files are found.
            ValueError: If the MATLAB file contains an unsupported structure.
        """
        files = self._discover_files()
        logger.info("Found {} MATLAB file(s) in {}", len(files), self.data_folder)
        t_start = time.perf_counter()

        per_file_data: list[NDArray] = []
        rows = cols = 0
        frequencies: NDArray | None = None
        for file in files:
            full_path = os.path.join(self.data_folder, file)
            stacked_data, rows, cols, frequencies = self._load_single_file(full_path)
            per_file_data.append(stacked_data)

        if frequencies is None:
            msg = "No frequency data found in MATLAB files."
            raise DataLoadError(msg)

        data_array = self._assemble_data_array(per_file_data, rows, cols, frequencies)

        elapsed = time.perf_counter() - t_start
        logger.info("MATLAB data loaded in {:.2f}s -- shape {}", elapsed, data_array.shape)
        return data_array

    def _discover_files(self: Self) -> list[str]:
        """Find run_*.mat files in the data folder, sorted by name.

        Raises:
            DataLoadError: If no matching files are found.
        """
        files = sorted(
            f for f in os.listdir(self.data_folder) if f.endswith(".mat") and "run_" in f
        )
        if not files:
            msg = "No valid MATLAB files found in the folder."
            raise DataLoadError(msg)
        return files

    def _load_single_file(self: Self, full_path: str) -> tuple[NDArray, int, int, NDArray]:
        """Load and parse one MATLAB file.

        Returns:
            Tuple of (stacked_data, rows, cols, frequencies) for this file.

        Raises:
            DataLoadError: If a required key is missing from the file.
        """
        logger.debug("Loading MATLAB file: {}", full_path)
        try:
            mat_data = mat73.loadmat(full_path)
            logger.debug("Loaded {} with mat73", full_path)
        except (TypeError, OSError):
            mat_data = loadmat(full_path)
            logger.debug("Loaded {} with scipy.io.loadmat (mat73 fallback)", full_path)

        stacked_data = self._process_mat_file(mat_data)
        logger.debug("Extracted data shape: {} from {}", stacked_data.shape, full_path)

        try:
            rows = int(np.squeeze(mat_data["imgNumRows"]))
            cols = int(np.squeeze(mat_data["imgNumCols"]))
        except KeyError as e:
            msg = f"Missing required key in MATLAB file: {e}"
            raise DataLoadError(msg) from e

        frequencies = self._parse_frequencies(mat_data)
        return stacked_data, rows, cols, frequencies

    @staticmethod
    def _parse_frequencies(mat_data: dict[str, Any]) -> NDArray:
        """Extract the frequency axis from one file's MATLAB dict.

        A concatenated ``freqList`` is split into (low, high) branches when
        ``numFreqs`` indicates it covers both ranges.

        Raises:
            DataLoadError: If the ``freqList`` key is missing.
        """
        try:
            freq_list = np.squeeze(mat_data["freqList"])
            n_freqs = int(np.squeeze(mat_data["numFreqs"])) if "numFreqs" in mat_data else None
        except KeyError as e:
            msg = f"Missing required key in MATLAB file: {e}"
            raise DataLoadError(msg) from e

        if n_freqs is not None and n_freqs != len(freq_list):
            return np.array([freq_list[:n_freqs], freq_list[n_freqs:]])
        return freq_list

    @staticmethod
    def _assemble_data_array(
        per_file_data: list[NDArray],
        rows: int,
        cols: int,
        frequencies: NDArray,
    ) -> xr.DataArray:
        """Stack per-file data into the final oriented ``xr.DataArray``."""
        # Stack polarity axis from multiple files
        if len(per_file_data) == 1:
            raw_data = per_file_data[0][np.newaxis, ...]
        else:
            raw_data = np.stack(per_file_data, axis=0)

        # raw_data shape: (n_pol, n_frange, n_pixels, n_freqs)
        n_pol, n_frange, _n_pixels, n_freqs = raw_data.shape

        # Reshape flattened pixels to 2D spatial grid
        raw_data = raw_data.reshape(n_pol, n_frange, rows, cols, n_freqs)
        logger.debug("Reshaped data to ({}, {}, {}, {}, {})", n_pol, n_frange, rows, cols, n_freqs)

        # The MATLAB instrument stores rows bottom-to-top (y=0 at bottom), but the
        # LED/laser camera images (CSV) use top-to-bottom convention (y=0 at top).
        # Flip the y axis so ODMR data aligns with the camera images.
        raw_data = np.ascontiguousarray(raw_data[:, :, ::-1, :, :])

        # Build frequency coordinate
        if frequencies.ndim == 1:
            freq_ghz = np.tile(frequencies, (n_frange, 1)) / 1e9
        else:
            freq_ghz = frequencies / 1e9
        logger.debug("Frequency range: {:.4f} - {:.4f} GHz", freq_ghz.min(), freq_ghz.max())

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
        logger.debug("Found {} image stacks in MATLAB file", n_img_stacks)
        if n_img_stacks == DUAL_POLARITY_STACKS:
            return np.stack([mat_file["imgStack1"].T, mat_file["imgStack2"].T], axis=0)
        if n_img_stacks == QUAD_POLARITY_STACKS:
            stack_low = np.concatenate([mat_file["imgStack1"], mat_file["imgStack2"]], axis=0).T
            stack_high = np.concatenate([mat_file["imgStack3"], mat_file["imgStack4"]], axis=0).T
            return np.stack([stack_low, stack_high], axis=0)
        msg = "Unsupported number of image stacks in MATLAB file."
        raise DataLoadError(msg)
