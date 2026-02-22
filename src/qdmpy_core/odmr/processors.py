"""Data processing pipeline for ODMR spectroscopy.

This module implements a flexible processing framework for ODMR data through
a collection of processor classes and a manager to coordinate them.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from qdmpy_core.constants import FLUORESCENCE_DELTA_THRESHOLD

if TYPE_CHECKING:
    from qdmpy_core.odmr.data import ODMRData


@runtime_checkable
class Processor(Protocol):
    """Protocol for ODMR data processors.

    Implement this protocol to add a custom processing step to the ODMR
    pipeline. Processors must be stateless with respect to data: ``process``
    receives a data object and returns a **new** data object without mutating
    the input.

    **Custom processor contract:**

    .. code-block:: python

        from qdmpy_core import Processor, ODMR, ODMRData

        class MyProcessor:
            def process(self, data: ODMRData) -> ODMRData:
                # Return a new ODMRData — never mutate the input.
                new_da = data.data * 1.05   # example: scale all values
                return ODMRData(data=new_da, metadata=data.metadata.copy())

            def describe(self) -> str:
                return 'MyProcessor(scale=1.05)'

        odmr = ODMR(odmr_data)
        odmr.processor_manager.add_processor(MyProcessor())
        odmr.process_data()

    Note:
        ``describe()`` is used for logging and pipeline inspection.
        Processors do not need to inherit from any base class — structural
        subtyping (duck typing) is used.
    """

    def process(self, data: ODMRData) -> ODMRData:
        """Process the given ODMRData and return a new instance.

        Args:
            data: Input ODMR data (treat as immutable).

        Returns:
            New ODMRData with the processing step applied.
        """
        ...

    def describe(self) -> str:
        """Return a human-readable description of this processor.

        Returns:
            String identifying the processor and its key parameters.
        """
        ...


class BaseProcessor(BaseModel):
    """Abstract base class for ODMR processors."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def process(self, data: ODMRData) -> ODMRData:
        """Process the given ODMRData instance and return a new instance."""

    def describe(self) -> str:
        """Return a human-readable description of this processor."""
        return f"{self.__class__.__name__}({self.model_dump()})"

    def to_config(self) -> dict[str, Any]:
        """Serialize this processor to a JSON-compatible dict."""
        return self.model_dump()


class NormalizationProcessor(BaseProcessor):
    """Normalizes ODMR data along the frequency dimension.

    Attributes:
        method: The normalization method to use (e.g., 'max').
    """

    type: Literal["NormalizationProcessor"] = "NormalizationProcessor"
    method: str = "max"

    def process(self, data: ODMRData) -> ODMRData:
        """Normalize the data based on the selected method."""
        from qdmpy_core.odmr.data import ODMRData

        logger.debug(f"Normalizing data using method: {self.method}")
        factors = self._get_norm_factors(data.data, self.method)
        normalized = data.data / factors
        return ODMRData(data=normalized, metadata=data.metadata.copy())

    def _get_norm_factors(self, da: xr.DataArray, method: str) -> xr.DataArray:
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

    type: Literal["BinningProcessor"] = "BinningProcessor"
    bin_factor: int = Field(gt=0)

    def process(self, data: ODMRData) -> ODMRData:
        """Bin the data spatially by the specified factor."""
        from qdmpy_core.odmr.data import ODMRData

        logger.debug(f"Binning data with factor: {self.bin_factor}")
        binned = data.data.coarsen(y=self.bin_factor, x=self.bin_factor, boundary="trim").mean()  # type: ignore[attr-defined]
        return ODMRData(data=binned, metadata=data.metadata.copy())


class OutlierProcessor(BaseProcessor):
    """Masks outlier values in ODMR data using z-scores along the frequency dimension.

    Attributes:
        z_score_threshold: The z-score threshold above which a value is considered an outlier.
    """

    type: Literal["OutlierProcessor"] = "OutlierProcessor"
    z_score_threshold: float = Field(default=0.003, gt=0)

    def process(self, data: ODMRData) -> ODMRData:
        """Apply an outlier mask based on the z-score threshold."""
        from qdmpy_core.odmr.data import ODMRData

        logger.debug(f"Masking outliers with z_score_threshold: {self.z_score_threshold}")
        data_mean = data.data.mean(dim="freq_idx")
        data_std = data.data.std(dim="freq_idx")
        z_scores = np.abs((data.data - data_mean) / (data_std + 1e-10))
        mask = z_scores > self.z_score_threshold
        processed = data.data.where(~mask)
        return ODMRData(data=processed, metadata=data.metadata.copy())


class FluorescenceCorrectionProcessor(BaseProcessor):
    """Global fluorescence correction for ODMR data.

    Attributes:
        correction_factor: The fluorescence correction factor.
    """

    type: Literal["FluorescenceCorrectionProcessor"] = "FluorescenceCorrectionProcessor"
    correction_factor: float = Field(default=0.2, gt=0)

    def process(self, data: ODMRData) -> ODMRData:
        """Apply fluorescence correction to the ODMR data."""
        from qdmpy_core.odmr.data import ODMRData

        logger.info(f"Applying fluorescence correction with factor: {self.correction_factor}")
        _, baseline_corrected = analyze_fluorescence_effects(data)
        correction = self.correction_factor * baseline_corrected
        processed = data.data - correction
        return ODMRData(data=processed, metadata=data.metadata.copy())


ProcessorSpec = Annotated[
    NormalizationProcessor | BinningProcessor | OutlierProcessor | FluorescenceCorrectionProcessor,
    Field(discriminator="type"),
]

_adapter: TypeAdapter[
    NormalizationProcessor | BinningProcessor | OutlierProcessor | FluorescenceCorrectionProcessor
] = TypeAdapter(ProcessorSpec)


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
            delta_copy[delta_copy > FLUORESCENCE_DELTA_THRESHOLD] = np.nan

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

    _f, ax = plt.subplots(
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


class ODMRProcessorManager:
    """Manages multiple processors for ODMR data.

    Attributes:
        processors: List of processors in the pipeline.
    """

    def __init__(self) -> None:
        """Initialize an empty processing pipeline."""
        self.processors: list[BaseProcessor] = []

    def add_processor(self, processor: BaseProcessor) -> None:
        """Add a processor to the processing pipeline."""
        logger.debug(f"Adding processor: {processor.__class__.__name__}")
        self.processors.append(processor)

    def process(self, data: ODMRData) -> ODMRData:
        """Apply all processors sequentially and record the pipeline config in metadata."""
        from qdmpy_core.odmr.data import ODMRData as _ODMRData

        logger.info("Starting processing pipeline.")
        pipeline_config = [
            p.to_config() if hasattr(p, "to_config") else {"describe": p.describe()}
            for p in self.processors
        ]
        for processor in self.processors:
            logger.debug(f"Applying processor: {processor.__class__.__name__}")
            data = processor.process(data)
        logger.info("Processing pipeline completed.")
        metadata = data.metadata.copy()
        metadata["pipeline"] = pipeline_config
        return _ODMRData(data=data.data, metadata=metadata)

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> ODMRProcessorManager:
        """Reconstruct a pipeline from a serialized config list."""
        manager = cls()
        for step in config:
            manager.add_processor(_adapter.validate_python(step))
        return manager

    @property
    def pipeline_config(self) -> list[dict[str, Any]]:
        """Current pipeline as a list of serializable config dicts."""
        return [
            p.to_config() if hasattr(p, "to_config") else {"describe": p.describe()}
            for p in self.processors
        ]

    def list_processors(self) -> list[str]:
        """List the type names of processors in the pipeline."""
        return [type(p).__name__ for p in self.processors]
