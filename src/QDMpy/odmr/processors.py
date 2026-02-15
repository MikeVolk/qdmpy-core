"""Data processing pipeline for ODMR spectroscopy.

This module implements a flexible processing framework for ODMR data through
a collection of processor classes and a manager to coordinate them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt
from typing_extensions import Self

if TYPE_CHECKING:
    from QDMpy.odmr.data import ODMRData


class BaseProcessor(ABC):
    """Abstract base class for ODMR processors."""

    @abstractmethod
    def process(self: Self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Process the given ODMRData instance and return a new instance."""


class NormalizationProcessor(BaseProcessor):
    """Normalizes ODMR data along the frequency dimension.

    Attributes:
        method: The normalization method to use (e.g., 'max').
    """

    def __init__(self: Self, method: str = "max") -> None:
        self.method = method

    def process(self: Self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Normalize the data based on the selected method."""
        from QDMpy.odmr.data import ODMRData

        logger.debug(f"Normalizing data using method: {self.method}")
        factors = self._get_norm_factors(data.data, self.method)
        normalized = data.data / factors
        metadata = data.metadata.copy()
        metadata["normalized"] = True
        return ODMRData(data=normalized, metadata=metadata)

    def _get_norm_factors(self: Self, da: xr.DataArray, method: str) -> xr.DataArray:
        """Calculate normalization factors."""
        if method == "max":
            return da.max(dim="freq_idx")
        raise NotImplementedError(
            f"Normalization method '{method}' is not implemented.",
        )


class BinningProcessor(BaseProcessor):
    """Spatial binning of ODMR data using xarray coarsen.

    Attributes:
        bin_factor: The factor by which to bin the data spatially.
    """

    def __init__(self: Self, bin_factor: int) -> None:
        if bin_factor <= 0:
            raise ValueError("Bin factor must be greater than 0.")
        self.bin_factor = bin_factor

    def process(self: Self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Bin the data spatially by the specified factor."""
        from QDMpy.odmr.data import ODMRData

        logger.debug(f"Binning data with factor: {self.bin_factor}")
        binned = data.data.coarsen(y=self.bin_factor, x=self.bin_factor, boundary="trim").mean()

        metadata = data.metadata.copy()
        metadata["binned"] = True
        metadata["bin_factor"] = self.bin_factor
        return ODMRData(data=binned, metadata=metadata)


class OutlierProcessor(BaseProcessor):
    """Masks outlier values in ODMR data using z-scores along the frequency dimension.

    Attributes:
        threshold: The threshold for outlier detection in standard deviations.
    """

    def __init__(self: Self, threshold: float = 0.001) -> None:
        self.threshold = threshold

    def process(self: Self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Apply an outlier mask based on the threshold."""
        from QDMpy.odmr.data import ODMRData

        logger.debug(f"Masking outliers with threshold: {self.threshold}")
        data_mean = data.data.mean(dim="freq_idx")
        data_std = data.data.std(dim="freq_idx")
        z_scores = np.abs((data.data - data_mean) / (data_std + 1e-10))
        mask = z_scores > (self.threshold * 3)
        processed = data.data.where(~mask)

        metadata = data.metadata.copy()
        metadata["outlier_masking"] = {"threshold": self.threshold}
        return ODMRData(data=processed, metadata=metadata)


class FluorescenceCorrectionProcessor(BaseProcessor):
    """Global fluorescence correction for ODMR data.

    Attributes:
        correction_factor: The fluorescence correction factor.
    """

    def __init__(self: Self, correction_factor: float = 0.2) -> None:
        self.correction_factor = correction_factor

    def process(self: Self, data: ODMRData, **kwargs: Any) -> ODMRData:
        """Apply fluorescence correction to the ODMR data."""
        from QDMpy.odmr.data import ODMRData

        factor = kwargs.get(
            "correction_factor", kwargs.get("glob_fluorescence", self.correction_factor)
        )
        logger.info(f"Applying fluorescence correction with factor: {factor}")

        _, baseline_corrected = analyze_fluorescence_effects(data)
        correction = factor * baseline_corrected
        processed = data.data - correction

        metadata = data.metadata.copy()
        if "fluorescence_correction" not in metadata:
            metadata["fluorescence_correction"] = {}
        metadata["fluorescence_correction"]["factor"] = factor
        metadata["fluorescence_correction"]["applied"] = True
        return ODMRData(data=processed, metadata=metadata)


def analyze_fluorescence_effects(
    data: ODMRData, pixel_idx: int | None = None
) -> tuple[int, xr.DataArray]:
    """Analyze the global fluorescence effects in ODMR data.

    Args:
        data: The input ODMR data.
        pixel_idx: Optional index of a specific pixel (in flattened y*x space).

    Returns:
        Tuple of (selected pixel flat index, baseline-corrected mean data).
    """
    values = data.data.values
    n_y = data.data.sizes["y"]
    n_x = data.data.sizes["x"]
    n_pixels = n_y * n_x

    # Flatten spatial dims for pixel selection
    flat_shape = (
        data.data.sizes["polarity"],
        data.data.sizes["freq_range"],
        n_pixels,
        data.data.sizes["freq_idx"],
    )
    flat_data = values.reshape(flat_shape)

    if pixel_idx is None:
        try:
            delta = np.nansum(
                np.square(flat_data - np.nanmean(flat_data, axis=2, keepdims=True)),
                axis=-1,
            )
            delta_copy = delta.copy()
            delta_copy[delta_copy > 0.001] = np.nan

            if np.all(np.isnan(delta_copy)):
                logger.warning("All values in delta_copy are NaN. Using middle pixel.")
                flat_idx = n_pixels // 2
            else:
                flat_idx = int(np.unravel_index(np.nanargmax(delta_copy), delta_copy.shape)[2])
            logger.info(f"Automatically selected pixel index: {flat_idx}")
        except ValueError:
            logger.warning("Error finding representative pixel. Using middle pixel.")
            flat_idx = n_pixels // 2
    else:
        flat_idx = int(pixel_idx)

    # Mean across all spatial pixels, keep as xarray for broadcasting
    mean_data = data.data.mean(dim=("y", "x"))

    # Baseline from off-resonance edges
    n_freqs = data.data.sizes["freq_idx"]
    n_edge = max(int(n_freqs * 0.05), 1)
    baseline_left = mean_data.isel(freq_idx=slice(0, n_edge)).mean(dim="freq_idx")
    baseline_right = mean_data.isel(freq_idx=slice(-n_edge, None)).mean(dim="freq_idx")
    baseline = (baseline_left + baseline_right) / 2

    baseline_corrected = mean_data - baseline

    return int(flat_idx), baseline_corrected


def preview_fluorescence_correction(
    data: ODMRData, correction_factor: float = 0.2, pixel_idx: int | None = None
) -> None:
    """Preview the effect of fluorescence correction on ODMR data."""
    idx_flat, baseline_corrected = analyze_fluorescence_effects(data, pixel_idx)
    correction = correction_factor * baseline_corrected

    n_pol = data.data.sizes["polarity"]
    n_frange = data.data.sizes["freq_range"]
    n_y = data.data.sizes["y"]
    n_x = data.data.sizes["x"]

    # Get pixel data in flat space
    flat_values = data.data.values.reshape(n_pol, n_frange, n_y * n_x, -1)

    f, ax = plt.subplots(
        n_pol,
        n_frange,
        sharex=False,
        sharey=True,
        figsize=(4 * n_frange, 3 * n_pol),
    )

    if n_pol == 1 and n_frange == 1:
        ax = np.array([[ax]])
    elif n_pol == 1:
        ax = np.array([ax])
    elif n_frange == 1:
        ax = np.array([ax]).T

    freq_ghz = data.data.coords["freq_ghz"].values

    for p in range(n_pol):
        for fr in range(n_frange):
            current_data = flat_values[p, fr, idx_flat].copy()
            freqs = freq_ghz[fr]
            corr_vals = correction.isel(polarity=p, freq_range=fr).values

            ax[p, fr].plot(freqs, current_data, "k.-", label="Original")
            ax[p, fr].plot(
                freqs,
                current_data - corr_vals,
                "r.-",
                label=f"Corrected (Factor={correction_factor})",
            )
            ax[p, fr].plot(freqs, 1 + corr_vals, "r--", alpha=0.5, label="Correction")

            polarity_label = {0: "+", 1: "-"}.get(p, f"P{p}")
            frange_label = {0: "Low", 1: "High"}.get(fr, f"F{fr}")
            ax[p, fr].set_title(f"Polarity: {polarity_label}, Frequency Range: {frange_label}")
            ax[p, fr].set_xlabel("Frequency [GHz]")
            ax[p, fr].set_ylabel("ODMR Contrast")
            ax[p, fr].legend()
            ax[p, fr].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f"Fluorescence Correction Preview (Pixel {idx_flat})", y=1.02)
    plt.show()


visualize_fluorescence_correction = preview_fluorescence_correction


class ODMRProcessorManager:
    """Manages multiple processors for ODMR data.

    Attributes:
        processors: List of processors in the pipeline.
    """

    def __init__(self: Self) -> None:
        self.processors: list[BaseProcessor] = []

    def add_processor(self: Self, processor: BaseProcessor) -> None:
        """Add a processor to the processing pipeline."""
        logger.debug(f"Adding processor: {processor.__class__.__name__}")
        self.processors.append(processor)

    def process(self: Self, data: ODMRData) -> ODMRData:
        """Apply all processors sequentially."""
        logger.info("Starting processing pipeline.")
        for processor in self.processors:
            logger.debug(f"Applying processor: {processor.__class__.__name__}")
            data = processor.process(data)
        logger.info("Processing pipeline completed.")
        return data

    def list_processors(self: Self) -> list[str]:
        """List the names of processors in the pipeline."""
        return [processor.__class__.__name__ for processor in self.processors]
