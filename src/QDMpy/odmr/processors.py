"""Data processing pipeline for ODMR spectroscopy.

This module implements a flexible processing framework for Optically Detected Magnetic
Resonance (ODMR) data through a collection of processor classes and a manager to
coordinate them. Key capabilities include:

- Modular processing: Individual processors for specific transformations
- Pipeline management: Sequential application of multiple processing steps
- Normalization: Various methods for normalizing spectral data
- Binning: Spatial and spectral data reduction techniques
- Outlier detection: Statistical identification of anomalous data points
- Fluorescence correction: Global and local background fluorescence compensation
- Metadata tracking: Preserving processing history throughout the pipeline

The processor architecture follows a strategy pattern, allowing flexible configuration
of data processing workflows while maintaining a consistent interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from skimage.measure import block_reduce

if TYPE_CHECKING:
    from QDMpy.odmr.data import ODMRData

LOG = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Abstract base class for ODMR processors.

    Each processor modifies an ODMRData instance and returns a new instance.
    """

    @abstractmethod
    def process(self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Process the given ODMRData instance and return a new instance.

        Args:
            data (ODMRData): The input data to process.
            **kwargs: Additional keyword arguments for specific processor implementations.

        Returns:
            ODMRData: A new instance containing the processed data.
        """


class NormalizationProcessor(BaseProcessor):
    """Handles normalization of ODMR data.

    Normalizes the frequency-dependent data (axis 3) for each pixel
    independently. This is typically used to normalize ODMR spectra
    to a consistent scale across all pixels.

    Attributes:
        method (str): The normalization method to use (e.g., 'max').
    """

    def __init__(self, method: str = "max") -> None:
        """Initialize the NormalizationProcessor.

        Args:
            method (str): Normalization method. Default is 'max'.
        """
        self.method = method

    def process(self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Normalize the data based on the selected method.

        Args:
            data (ODMRData): The input data to normalize.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            ODMRData: A new instance containing the normalized data.

        Raises:
            NotImplementedError: If the specified normalization method is not supported.
        """
        LOG.debug("Normalizing data using method: %s", self.method)
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
        """Calculate normalization factors based on the selected method.

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
            f"Normalization method '{method}' is not implemented.",
        )


class BinningProcessor(BaseProcessor):
    """Handles spatial binning of ODMR data.

    Attributes:
        bin_factor (int): The factor by which to bin the data spatially.
    """

    def __init__(self, bin_factor: int) -> None:
        """Initialize the BinningProcessor.

        Args:
            bin_factor (int): The spatial binning factor. Must be > 0.

        Raises:
            ValueError: If the bin factor is less than or equal to 0.
        """
        if bin_factor <= 0:
            raise ValueError("Bin factor must be greater than 0.")
        self.bin_factor = bin_factor

    def process(self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Bin the data by the specified factor.

        This method takes the raw data with shape (channels, runs, pixels, frequencies)
        and performs spatial binning on the pixels. It reshapes the flattened pixels
        into a 2D image using scan_dimensions before applying binning, then flattens
        the result back to the original data format.

        Args:
            data (ODMRData): The input data to bin with shape (channels, runs, pixels, frequencies).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            ODMRData: A new instance containing the binned data with reduced spatial resolution
                     but the same overall shape structure.
        """
        LOG.debug("Binning data with factor: %s", self.bin_factor)
        # Calculate spatial dimensions, ensuring compatibility with non-square images
        total_pixels = data.data.shape[2]
        # Try to determine rows and cols from scan_dimensions if available
        has_valid_dimensions = (
            hasattr(data, "scan_dimensions")
            and data.scan_dimensions is not None
            and len(data.scan_dimensions) == 2
        )
        if has_valid_dimensions:
            rows, cols = data.scan_dimensions
        else:
            # Fallback to assuming square images if dimensions are not available
            int(total_pixels**0.5)
            LOG.warning("Assuming square image for binning. Using scan_dimensions is recommended.")

        reshape_data = data.data.reshape(
            data.data.shape[0],
            data.data.shape[1],
            rows,
            cols,
            data.data.shape[-1],
        )

        binned = block_reduce(
            reshape_data,
            block_size=(1, 1, self.bin_factor, self.bin_factor, 1),
            func=np.nanmean,
        )
        binned = binned.reshape(
            data.data.shape[0],
            data.data.shape[1],
            -1,
            data.data.shape[-1],
        )
        metadata = data.metadata.copy()
        metadata["binned"] = True
        metadata["bin_factor"] = self.bin_factor

        new_scan_dimensions = (
            int(data.scan_dimensions[0] / self.bin_factor),
            int(data.scan_dimensions[1] / self.bin_factor),
        )

        return data.__class__(
            data=binned,
            scan_dimensions=new_scan_dimensions,
            frequencies=data.frequencies,
            metadata=metadata,
        )


class OutlierProcessor(BaseProcessor):
    """Handles masking of outliers in ODMR data.

    Identifies and masks outlier values in the ODMR spectra based on
    z-scores computed across the frequency dimension (axis 3). Values
    that exceed the threshold are replaced with NaN values.

    Attributes:
        threshold (float): The threshold for outlier detection in standard deviations.
    """

    def __init__(self, threshold: float = 0.001) -> None:
        """Initialize the OutlierProcessor.

        Args:
            threshold (float): Threshold for masking outliers.
        """
        self.threshold = threshold

    def process(self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Apply an outlier mask based on the threshold.

        Args:
            data (ODMRData): The input data to process.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            ODMRData: A new instance with outliers masked.
        """
        LOG.debug("Masking outliers with threshold: %s", self.threshold)
        # Use a more robust algorithm that considers standard deviation
        data_mean = np.mean(data.data, axis=-1, keepdims=True)
        data_std = np.std(data.data, axis=-1, keepdims=True)
        # Add small epsilon to avoid division by zero
        z_scores = np.abs((data.data - data_mean) / (data_std + 1e-10))
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


class FluorescenceCorrectionProcessor(BaseProcessor):
    """Handles global fluorescence correction for ODMR data.

    The global fluorescence correction compensates for systematic fluorescence
    variations that affect the baseline of ODMR measurements. It applies a
    correction factor based on the difference between the mean ODMR signal and
    its baseline.

    Attributes:
        correction_factor (float): The fluorescence correction factor.
    """

    def __init__(self, correction_factor: float = 0.2) -> None:
        """Initialize the FluorescenceCorrectionProcessor.

        Args:
            correction_factor (float): The fluorescence correction factor.
                Higher values apply a stronger correction. Default is 0.2.
        """
        self.correction_factor = correction_factor

    def process(self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Apply fluorescence correction to the ODMR data.

        Args:
            data (ODMRData): The input data to process.
            **kwargs: Additional keyword arguments.
                correction_factor (float, optional): Override the fluorescence
                    correction factor. If provided, will be used instead
                    of the instance's correction_factor.
                glob_fluorescence (float, optional): Legacy parameter name,
                    same as correction_factor.

        Returns:
            ODMRData: A new instance with fluorescence correction applied.
        """
        # Check for override correction_factor from kwargs (support both names for backward compatibility)
        factor = kwargs.get(
            "correction_factor", kwargs.get("glob_fluorescence", self.correction_factor)
        )

        LOG.info("Applying fluorescence correction with factor: %s", factor)

        # Get the baseline-corrected data
        _, baseline_corrected = analyze_fluorescence_effects(data)

        # Apply correction factor
        correction = factor * baseline_corrected

        # Apply correction
        processed_data = data.data.copy() - correction

        # Create new ODMRData instance with corrected data
        metadata = data.metadata.copy()
        if "fluorescence_correction" not in metadata:
            metadata["fluorescence_correction"] = {}

        metadata["fluorescence_correction"]["factor"] = factor
        metadata["fluorescence_correction"]["applied"] = True

        return data.__class__(
            data=processed_data,
            scan_dimensions=data.scan_dimensions,
            frequencies=data.frequencies,
            metadata=metadata,
        )


def analyze_fluorescence_effects(
    data: ODMRData, pixel_idx: Optional[int] = None
) -> Tuple[int, NDArray[np.float64]]:
    """Analyze the global fluorescence effects in ODMR data.

    This function evaluates fluorescence variations in ODMR data by identifying
    representative pixels and calculating global correction factors. It helps identify
    suitable correction parameters without modifying the data.

    Args:
        data (ODMRData): The input ODMR data.
        pixel_idx (Optional[int], optional): The index of the specific pixel to analyze.
            If None, the function will automatically select a representative pixel.
            Default is None.

    Returns:
        Tuple[int, NDArray[np.float64]]: A tuple containing:
            - The index of the analyzed pixel
            - The calculated baseline-corrected mean data used for correction
    """
    if pixel_idx is None:
        try:
            # Find the most divergent pixel from mean (but not extreme outliers)
            delta = np.nansum(
                np.square(data.data - np.nanmean(data.data, axis=2, keepdims=True)), axis=-1
            )
            delta_copy = delta.copy()
            delta_copy[delta_copy > 0.001] = (
                np.nan
            )  # Mask high values to find a representative pixel

            # Check if all values are NaN
            if np.all(np.isnan(delta_copy)):
                LOG.warning("All values in delta_copy are NaN. Using middle pixel instead.")
                flat_idx = data.data.shape[2] // 2  # Use middle pixel as fallback
            else:
                flat_idx = int(np.unravel_index(np.nanargmax(delta_copy), delta_copy.shape)[2])

            LOG.info("Automatically selected pixel index: %s", flat_idx)
        except ValueError:
            # Fallback to middle pixel if any error occurs
            LOG.warning("Error finding representative pixel. Using middle pixel instead.")
            flat_idx = data.data.shape[2] // 2
    else:
        flat_idx = int(pixel_idx)

    # Calculate the mean ODMR across all pixels
    mean_data = np.nanmean(data.data, axis=2, keepdims=True)

    # Calculate baseline (off-resonance regions)
    n_freqs = mean_data.shape[-1]
    idx_left = slice(0, max(int(n_freqs * 0.05), 1))
    idx_right = slice(-max(int(n_freqs * 0.05), 1), None)

    baseline_left_mean = np.nanmean(mean_data[..., idx_left], axis=-1)
    baseline_right_mean = np.nanmean(mean_data[..., idx_right], axis=-1)
    baseline_mean = (baseline_left_mean + baseline_right_mean) / 2

    # Calculate correction: (mean_data - baseline)
    baseline_corrected = mean_data - baseline_mean[..., np.newaxis]

    return int(flat_idx), baseline_corrected


def preview_fluorescence_correction(
    data: ODMRData, correction_factor: float = 0.2, pixel_idx: Optional[int] = None
) -> None:
    """Preview the effect of fluorescence correction on ODMR data.

    This function creates a plot showing the original data, the corrected data,
    and the correction factor for a specific pixel. It helps users understand
    and evaluate the impact of fluorescence correction before actually applying it.

    Args:
        data (ODMRData): The input ODMR data.
        correction_factor (float, optional): The fluorescence correction factor
            to visualize. Default is 0.2.
        pixel_idx (Optional[int], optional): The index of the pixel to visualize.
            If None, the function will automatically select a representative pixel.
            Default is None.
    """
    # Get representative pixel and baseline-corrected data
    idx_flat, baseline_corrected = analyze_fluorescence_effects(data, pixel_idx)

    # Calculate correction by applying the correction factor
    correction = correction_factor * baseline_corrected

    # Create plot
    f, ax = plt.subplots(
        data.data.shape[0],
        data.data.shape[1],
        sharex=False,
        sharey=True,
        figsize=(4 * data.data.shape[1], 3 * data.data.shape[0]),
    )

    # Handle case with single subplot
    if data.data.shape[0] == 1 and data.data.shape[1] == 1:
        ax = np.array([[ax]])
    elif data.data.shape[0] == 1:
        ax = np.array([ax])
    elif data.data.shape[1] == 1:
        ax = np.array([ax]).T

    # Get frequency values in GHz for plotting
    frequencies = (
        data.frequencies / 1e9 if hasattr(data, "frequencies") else np.arange(data.data.shape[-1])
    )

    # Plot for each polarity and frequency range
    for p in range(data.data.shape[0]):  # polarities
        for f in range(data.data.shape[1]):  # frequency ranges
            # Get current data for this pixel
            current_data = data.data[p, f, idx_flat].copy()

            # Plot current data
            (line,) = ax[p, f].plot(
                frequencies[f]
                if isinstance(frequencies, np.ndarray) and len(frequencies.shape) > 1
                else frequencies,
                current_data,
                "k.-",
                label="Original"
                if "fluorescence_correction" not in data.metadata
                else f'Current (Factor={data.metadata["fluorescence_correction"].get("factor", 0)})',
            )

            # Plot data with the new correction applied
            ax[p, f].plot(
                frequencies[f]
                if isinstance(frequencies, np.ndarray) and len(frequencies.shape) > 1
                else frequencies,
                current_data - correction[p, f, 0],
                "r.-",
                label=f"Corrected (Factor={correction_factor})",
            )

            # Plot correction factor
            ax[p, f].plot(
                frequencies[f]
                if isinstance(frequencies, np.ndarray) and len(frequencies.shape) > 1
                else frequencies,
                1 + correction[p, f, 0],
                "r--",
                alpha=0.5,
                label="Correction",
            )

            # Set titles and labels
            polarity_label = {0: "+", 1: "-"}.get(p, f"P{p}")
            frange_label = {0: "Low", 1: "High"}.get(f, f"F{f}")
            ax[p, f].set_title(f"Polarity: {polarity_label}, Frequency Range: {frange_label}")
            ax[p, f].set_xlabel("Frequency [GHz]")
            ax[p, f].set_ylabel("ODMR Contrast")
            ax[p, f].legend()
            ax[p, f].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f"Fluorescence Correction Preview (Pixel {idx_flat})", y=1.02)
    plt.show()


# Alias for backward compatibility
visualize_fluorescence_correction = preview_fluorescence_correction


class ODMRProcessorManager:
    """Manages multiple processors for ODMR data.

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
        self.processors: list[BaseProcessor] = []

    def add_processor(self, processor: BaseProcessor) -> None:
        """Add a processor to the processing pipeline.

        Args:
            processor (BaseProcessor): An instance of a processor to add.
        """
        LOG.debug("Adding processor: %s", processor.__class__.__name__)
        self.processors.append(processor)

    def process(self, data: ODMRData) -> ODMRData:
        """Apply all processors sequentially to the given ODMRData instance.

        Args:
            data (ODMRData): The input data to process.

        Returns:
            ODMRData: A new ODMRData instance containing the processed data.
        """
        LOG.info("Starting processing pipeline.")
        for processor in self.processors:
            LOG.debug("Applying processor: %s", processor.__class__.__name__)
            data = processor.process(data)
        LOG.info("Processing pipeline completed.")
        return data

    def list_processors(self) -> list[str]:
        """List the names of processors in the pipeline.

        Returns:
            List[str]: A list of processor class names in the order they were added.
        """
        return [processor.__class__.__name__ for processor in self.processors]
