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
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from qdmpy.constants import FLUORESCENCE_DELTA_THRESHOLD

if TYPE_CHECKING:
    from qdmpy.odmr.data import ODMRData


@runtime_checkable
class Processor(Protocol):
    """Protocol for ODMR data processors.

    Implement this protocol to add a custom processing step to the ODMR
    pipeline. Processors must be stateless with respect to data: ``process``
    receives a data object and returns a **new** data object without mutating
    the input.

    **Custom processor contract:**

    .. code-block:: python

        from qdmpy import Processor, ODMR, ODMRData

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
    """Normalizes ODMR data by dividing each pixel's spectrum by its mean intensity.

    Mean-normalization preserves per-pixel off-resonance baseline variation, which
    is required for downstream fluorescence correction. Max-normalization forces
    every pixel's off-resonance level to exactly 1.0, destroying this information,
    and is therefore not supported.

    Attributes:
        method: The normalization method. Only ``'mean'`` is supported.
    """

    type: Literal["NormalizationProcessor"] = "NormalizationProcessor"
    method: str = "mean"

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Reject max-normalization and unknown methods at construction time."""
        if v == "max":
            raise ValueError(
                "method='max' is not physically valid: max-normalization forces every "
                "pixel's off-resonance level to exactly 1.0, destroying the per-pixel "
                "baseline variation required for fluorescence correction. "
                "Replace with method='mean'."
            )
        if v != "mean":
            raise ValueError(f"Unsupported normalization method '{v}'. Only 'mean' is supported.")
        return v

    def process(self, data: ODMRData) -> ODMRData:
        """Normalize the data by the per-pixel mean across the frequency dimension."""
        from qdmpy.odmr.data import ODMRData

        logger.debug(f"Normalizing data using method: {self.method}")
        factors = data.data.mean(dim="freq_idx")
        normalized = data.data / factors
        return ODMRData(data=normalized, metadata=data.metadata.copy())


class BinningProcessor(BaseProcessor):
    """Spatial binning of ODMR data using xarray coarsen.

    Attributes:
        bin_factor: The factor by which to bin the data spatially.
    """

    type: Literal["BinningProcessor"] = "BinningProcessor"
    bin_factor: int = Field(gt=0)

    def process(self, data: ODMRData) -> ODMRData:
        """Bin the data spatially by the specified factor."""
        from qdmpy.odmr.data import ODMRData

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
        from qdmpy.odmr.data import ODMRData

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
        from qdmpy.odmr.data import ODMRData

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
    from qdmpy.plotting import plot_fluorescence_correction

    plot_fluorescence_correction(data, correction_factor, pixel_idx)


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
        from qdmpy.odmr.data import ODMRData as _ODMRData

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
