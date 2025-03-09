"""Module: QDMpy.odmr.io.
=====================

This module provides loader classes to handle the input of ODMR data from different
sources, such as MATLAB files. It includes an abstract base class `BaseLoader` and
a concrete implementation `MatlabLoader`.

Classes:
    - BaseLoader: Abstract base class for ODMR data loaders.
    - MatlabLoader: A loader for ODMR data from MATLAB files.

Imports:
    - Python standard library: os
    - Third-party: mat73, scipy.io (loadmat), numpy
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import mat73
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat


class BaseLoader(ABC):
    """Abstract base class for ODMR data loaders.

    Subclasses should implement the `load` method to provide specific functionality
    for loading ODMR data from different file types or sources.

    The loaded data should be in a standardized format with 4 dimensions:
    - Axis 0: Different polarities of measurements (typically 2 for positive/negative)
    - Axis 1: Different frequency ranges scanned in the experiment
    - Axis 2: Frequency points (number of frequency measurements per pixel)
    - Axis 3: Spatial pixels (flattened from a 2D image with rows x cols pixels)
    """

    @abstractmethod
    def load(self, **kwargs: Any) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
        """Load ODMR data.

        Returns:
            Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]: A tuple containing:
                - raw_data: The raw data array with shape (polarities, frequency_ranges, pixels, frequencies).
                - scan_dimensions: The scan dimensions (rows, cols) used to reshape pixels to 2D.
                - frequencies: The 1D array of frequencies used in the measurements.
        """


class MatlabLoader(BaseLoader):
    """Loader for ODMR data from MATLAB files.

    This loader supports both `.mat` files handled by `scipy.io.loadmat` and
    `.mat` files with modern structures handled by `mat73.loadmat`.

    Attributes:
        data_folder (str): Path to the folder containing MATLAB files.
    """

    def __init__(self, data_folder: str) -> None:
        """Initialize the MatlabLoader.

        Args:
            data_folder (str): Path to the folder containing MATLAB files.
        """
        self.data_folder = data_folder

    def load(self, **kwargs: Any) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
        """Load ODMR data from the specified folder.

        Args:
            kwargs (Any): Additional arguments for loading data (optional).

        Returns:
            Tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]: A tuple containing:
                - raw_data: The raw data array with shape (polarities, frequency_ranges, pixels, frequencies).
                  - Polarity axis (0): Contains measurements with different polarities (typically 2)
                  - Frequency ranges axis (1): Contains data from different frequency ranges
                  - Frequencies axis (2): Contains measurements at different frequencies
                  - Pixels axis (3): Contains flattened spatial pixels
                - scan_dimensions: The scan dimensions (rows, cols) used to reshape pixels to 2D.
                - frequencies: The 1D array of frequencies used in the measurements.
                - Note: All return values may be None if no files are processed.

        Raises:
            FileNotFoundError: If no valid MATLAB files are found in the folder.
            ValueError: If the MATLAB file contains an unsupported structure.
        """
        files = [
            f
            for f in os.listdir(self.data_folder)
            if f.endswith('.mat') and 'run_' in f
        ]
        if not files:
            raise FileNotFoundError('No valid MATLAB files found in the folder.')

        raw_data, img_shape, frequencies = None, None, None

        for file in files:
            full_path = os.path.join(self.data_folder, file)
            # Try using mat73 for v7.3 format files first, then fall back to loadmat
            try:
                mat_data = mat73.loadmat(full_path)
            except Exception:
                mat_data = loadmat(full_path)

            # Process MATLAB data into raw data arrays
            stacked_data = self._process_mat_file(mat_data)
            raw_data = (
                stacked_data
                if raw_data is None
                else np.stack((raw_data, stacked_data), axis=0)
            )

            try:
                img_shape = np.array(
                    [
                        int(np.squeeze(mat_data['imgNumRows'])),
                        int(np.squeeze(mat_data['imgNumCols'])),
                    ],
                )
            except KeyError as e:
                raise ValueError(f'Missing required key in MATLAB file: {e}')

            try:
                # Keep original dtype for frequencies to maintain precision
                frequencies = np.squeeze(mat_data['freqList'])
            except KeyError as e:
                raise ValueError(f'Missing required key in MATLAB file: {e}')

        return raw_data, img_shape, frequencies

    @staticmethod
    def _process_mat_file(mat_file: dict[str, Any]) -> NDArray:
        """Process a MATLAB file to extract raw ODMR data.

        Args:
            mat_file (dict[str, Any]): The MATLAB file content as a dictionary.

        Returns:
            NDArray: Processed raw data with shape (polarities, frequency_ranges, pixels, frequencies).
                - For 2 image stacks: Returns shape (2, frequency_ranges, pixels, frequencies)
                  where the 2 represents positive/negative polarities.
                - For 4 image stacks: Concatenates stacks 1+2 and 3+4 along the frequency ranges
                  axis, returning shape (2, frequency_ranges, pixels, frequencies).

        Raises:
            ValueError: If the MATLAB file contains an unsupported number of image stacks.
        """
        n_img_stacks = len([k for k in mat_file if 'imgStack' in k])
        if n_img_stacks == 2:
            return np.stack([mat_file['imgStack1'].T, mat_file['imgStack2'].T], axis=0)
        if n_img_stacks == 4:
            return np.stack(
                [
                    np.concatenate(
                        [mat_file['imgStack1'].T, mat_file['imgStack2'].T], axis=0,
                    ),
                    np.concatenate(
                        [mat_file['imgStack3'].T, mat_file['imgStack4'].T], axis=0,
                    ),
                ],
                axis=0,
            )
        raise ValueError('Unsupported number of image stacks in MATLAB file.')
