"""
Module: QDMpy.odmr.processors
=============================

This module provides various processors for modifying ODMR (Optically Detected Magnetic
Resonance) data. Each processor implements specific data processing functionality, and
the `ODMRProcessorManager` orchestrates the application of multiple processors.

Classes:
    - BaseProcessor: Abstract base class for defining data processors.
    - NormalizationProcessor: Handles normalization of ODMR data.
    - BinningProcessor: Handles spatial binning of ODMR data.
    - OutlierProcessor: Masks outliers in ODMR data.
    - ODMRProcessorManager: Manages and applies a sequence of processors.

Imports:
    - Python standard library: logging
    - Third-party: numpy, skimage.measure.block_reduce
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import List, TYPE_CHECKING, Any
import numpy as np
import logging
from skimage.measure import block_reduce

if TYPE_CHECKING:
    from QDMpy.odmr.data import ODMRData

LOG = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for ODMR processors.

    Each processor modifies an ODMRData instance and returns a new instance.
    """

    @abstractmethod
    def process(self, data: "ODMRData", **kwargs: Any) -> "ODMRData":
        """
        Process the given ODMRData instance and return a new instance.

        Args:
            data (ODMRData): The input data to process.
            **kwargs: Additional keyword arguments for specific processor implementations.

        Returns:
            ODMRData: A new instance containing the processed data.
        """
        pass


class NormalizationProcessor(BaseProcessor):
    """
    Handles normalization of ODMR data.

    Attributes:
        method (str): The normalization method to use (e.g., 'max').
    """

    def __init__(self, method: str = "max") -> None:
        """
        Initialize the NormalizationProcessor.

        Args:
            method (str): Normalization method. Default is 'max'.
        """
        self.method = method

    def process(self, data: "ODMRData", **kwargs: Any) -> "ODMRData":
        """
        Normalize the data based on the selected method.

        Args:
            data (ODMRData): The input data to normalize.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            ODMRData: A new instance containing the normalized data.

        Raises:
            NotImplementedError: If the specified normalization method is not supported.
        """
        LOG.debug(f"Normalizing data using method: {self.method}")
        factors = self._get_norm_factors(data.data, self.method)
        normalized_data = data.data / factors
        metadata = data.metadata.copy()
        metadata["normalized"] = True
        return data.__class__(
            data=normalized_data,
            scan_dimensions=data.scan_dimensions,
            frequencies=data.frequencies,
            metadata=metadata,
        )

    def _get_norm_factors(self, data: NDArray, method: str) -> NDArray:
        """
        Calculate normalization factors based on the selected method.

        Args:
            data (NDArray): The data to normalize.
            method (str): The normalization method.

        Returns:
            NDArray: The normalization factors.

        Raises:
            NotImplementedError: If the specified method is not supported.
        """
        if method == "max":
            return np.expand_dims(np.max(data, axis=-1), axis=-1)
        raise NotImplementedError(
            f"Normalization method '{method}' is not implemented."
        )


class BinningProcessor(BaseProcessor):
    """
    Handles spatial binning of ODMR data.

    Attributes:
        bin_factor (int): The factor by which to bin the data spatially.
    """

    def __init__(self, bin_factor: int) -> None:
        """
        Initialize the BinningProcessor.

        Args:
            bin_factor (int): The spatial binning factor. Must be > 0.

        Raises:
            ValueError: If the bin factor is less than or equal to 0.
        """
        if bin_factor <= 0:
            raise ValueError("Bin factor must be greater than 0.")
        self.bin_factor = bin_factor

    def process(self, data: "ODMRData", **kwargs: Any) -> "ODMRData":
        """
        Bin the data by the specified factor.

        Args:
            data (ODMRData): The input data to bin.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            ODMRData: A new instance containing the binned data.
        """
        LOG.debug(f"Binning data with factor: {self.bin_factor}")
        # Calculate spatial dimensions, ensuring compatibility with non-square images
        total_pixels = data.data.shape[2]
        # Try to determine rows and cols from scan_dimensions if available
        if hasattr(data, 'scan_dimensions') and data.scan_dimensions is not None and len(data.scan_dimensions) == 2:
            rows, cols = data.scan_dimensions
        else:
            # Fallback to assuming square images if dimensions are not available
            rows = cols = int(total_pixels ** 0.5)
            LOG.warning("Assuming square image for binning. Using scan_dimensions is recommended.")
            
        reshape_data = data.data.reshape(
            -1,
            rows,
            cols,
            data.data.shape[-1],
        )
        binned = block_reduce(
            reshape_data,
            block_size=(1, self.bin_factor, self.bin_factor, 1),
            func=np.nanmean,
        )
        binned = binned.reshape(
            data.data.shape[0], data.data.shape[1], -1, data.data.shape[-1]
        )
        metadata = data.metadata.copy()
        metadata["binned"] = True
        metadata["bin_factor"] = self.bin_factor
        return data.__class__(
            data=binned,
            scan_dimensions=data.scan_dimensions,
            frequencies=data.frequencies,
            metadata=metadata,
        )


class OutlierProcessor(BaseProcessor):
    """
    Handles masking of outliers in ODMR data.

    Attributes:
        threshold (float): The threshold for outlier detection.
    """

    def __init__(self, threshold: float = 0.001) -> None:
        """
        Initialize the OutlierProcessor.

        Args:
            threshold (float): Threshold for masking outliers.
        """
        self.threshold = threshold

    def process(self, data: "ODMRData", **kwargs: Any) -> "ODMRData":
        """
        Apply an outlier mask based on the threshold.

        Args:
            data (ODMRData): The input data to process.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            ODMRData: A new instance with outliers masked.
        """
        LOG.debug(f"Masking outliers with threshold: {self.threshold}")
        # Use a more robust algorithm that considers standard deviation
        data_mean = np.mean(data.data, axis=-1, keepdims=True)
        data_std = np.std(data.data, axis=-1, keepdims=True)
        z_scores = np.abs((data.data - data_mean) / (data_std + 1e-10))  # Add small epsilon to avoid div by zero
        mask = z_scores > (self.threshold * 3)  # Convert threshold to number of std deviations
        processed_data = data.data.copy()
        processed_data[mask] = np.nan
        metadata = data.metadata.copy()
        metadata["outlier_masking"] = {"threshold": self.threshold}
        return data.__class__(
            data=processed_data,
            scan_dimensions=data.scan_dimensions,
            frequencies=data.frequencies,
            metadata=metadata,
        )


class ODMRProcessorManager:
    """
    Manages multiple processors for ODMR data.

    Tracks and applies processing steps sequentially to transform ODMRData objects.

    Attributes:
        processors (List[BaseProcessor]): List of processors in the pipeline.

    Methods:
        add_processor: Add a processor to the pipeline.
        process: Apply all processors sequentially to data.
        list_processors: Get names of all processors in the pipeline.
    """

    def __init__(self) -> None:
        """Initialize an empty processor manager."""
        self.processors: List[BaseProcessor] = []

    def add_processor(self, processor: BaseProcessor) -> None:
        """
        Add a processor to the processing pipeline.

        Args:
            processor (BaseProcessor): An instance of a processor to add.
        """
        LOG.debug(f"Adding processor: {processor.__class__.__name__}")
        self.processors.append(processor)

    def process(self, data: "ODMRData") -> "ODMRData":
        """
        Apply all processors sequentially to the given ODMRData instance.

        Args:
            data (ODMRData): The input data to process.

        Returns:
            ODMRData: A new ODMRData instance containing the processed data.
        """
        LOG.info("Starting processing pipeline.")
        for processor in self.processors:
            LOG.debug(f"Applying processor: {processor.__class__.__name__}")
            data = processor.process(data)
        LOG.info("Processing pipeline completed.")
        return data

    def list_processors(self) -> List[str]:
        """
        List the names of processors in the pipeline.

        Returns:
            List[str]: A list of processor class names in the order they were added.
        """
        return [processor.__class__.__name__ for processor in self.processors]
