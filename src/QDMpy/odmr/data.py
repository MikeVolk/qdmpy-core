"""
Module: QDMpy.odmr.data
=======================

This module provides the `ODMRData` class for representing raw and processed ODMR
(Optically Detected Magnetic Resonance) data. It supports initialization from
raw data arrays or through a loader interface.

Classes:
    ODMRData: A class to encapsulate and manage ODMR data, including its raw data,
              scan dimensions, frequencies, and associated metadata.

Imports:
    - Python standard library: logging, sys, os
    - Third-party: numpy.typing (NDArray)
    - Local: QDMpy.odmr.io (MatlabLoader)
"""

from __future__ import annotations

from typing import Optional, Any, Dict, TYPE_CHECKING
from numpy.typing import NDArray
import logging
import sys
import os

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.insert(0, project_root)


if TYPE_CHECKING:
    from QDMpy.odmr.io import BaseLoader

LOG = logging.getLogger(__name__)


class ODMRData:
    """
    Represents raw and processed ODMR (Optically Detected Magnetic Resonance) data.

    Attributes:
        data (NDArray): The raw ODMR data, as a 4D numpy array with shape:
            - Axis 0: Data channels (typically 2 channels representing different measurements)
            - Axis 1: Runs or configurations (number of experimental runs)
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the ODMRData object.

        Args:
            data (NDArray): Raw ODMR data as a 4D numpy array with shape:
                - Axis 0: Data channels (typically 2 channels representing different measurements)
                - Axis 1: Runs or configurations (number of experimental runs)
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
        loader: "BaseLoader",
        loader_args: Optional[Dict[str, Any]] = None,
    ) -> "ODMRData":
        """
        Create an ODMRData instance using a loader.

        Args:
            loader (BaseLoader): An instantiated loader to fetch data dynamically.
            loader_args (Optional[Dict[str, Any]]): Arguments to pass to the loader
                                                    (optional).

        Returns:
            ODMRData: An instance populated with data loaded from the loader.

        Raises:
            RuntimeError: If the loader fails to fetch data.
        """
        LOG.info(f"Loading ODMR data using loader: {loader.__class__.__name__}")
        try:
            raw_data, scan_dimensions, frequencies = loader.load(**(loader_args or {}))
            return cls(raw_data, scan_dimensions, frequencies)
        except Exception as e:
            LOG.error(
                f"Failed to load data using loader {loader.__class__.__name__}: {e}"
            )
            raise RuntimeError(f"Data loading failed: {e}")

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the raw ODMR data.

        Returns:
            tuple[int, ...]: The shape of the raw data as a tuple (e.g., (rows, cols)).
        """
        return self.data.shape
