"""Data processing pipeline for ODMR spectroscopy.

This module implements a flexible processing framework for ODMR data through
a collection of processor classes and a manager to coordinate them.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
        subtyping (duck typing) is used. This is enough for direct pipeline
        use via ``add_processor()``. A processor that also needs to
        round-trip through ``ODMRProcessorManager.to_config()`` /
        ``from_config()`` (e.g. saving and reloading a pipeline) must
        additionally be a Pydantic ``BaseModel`` with a ``type:
        Literal[...]`` field, and register itself via
        :class:`ProcessorRegistry`'s ``@ProcessorRegistry.register``
        decorator.
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


class ProcessorRegistry:
    """Registry mapping a processor's ``type`` tag to its class (Open/Closed).

    A processor used only via ``add_processor()``/``process()`` needs no
    base class at all -- see the ``Processor`` protocol's custom-processor
    contract above. A processor that also wants to round-trip through
    ``ODMRProcessorManager.to_config()`` / ``from_config()`` (e.g. saving and
    reloading a pipeline) must additionally be a Pydantic ``BaseModel`` with
    a ``type: Literal[...]`` field, and register itself via
    ``@ProcessorRegistry.register`` -- mirroring
    :class:`~qdmpy.fitting.models.ModelRegistry`, this repo's established
    pattern for extending a closed set of types without editing existing
    code.

    Example:
        >>> @ProcessorRegistry.register
        ... class MyProcessor(BaseProcessor):
        ...     type: Literal['MyProcessor'] = 'MyProcessor'
        ...     def process(self, data): ...
    """

    _registry: ClassVar[dict[str, type[BaseProcessor]]] = {}

    @classmethod
    def register(
        cls: type[ProcessorRegistry], processor_cls: type[BaseProcessor]
    ) -> type[BaseProcessor]:
        """Register a processor class (usable as a decorator).

        Args:
            processor_cls: A ``BaseProcessor`` subclass whose ``type``
                field's default is used as the registry key.

        Returns:
            The processor class, unchanged.
        """
        type_name = processor_cls.model_fields["type"].default
        cls._registry[type_name] = processor_cls
        logger.debug("Registered processor type: {}", type_name)
        return processor_cls

    @classmethod
    def get(cls: type[ProcessorRegistry], type_name: str) -> type[BaseProcessor]:
        """Get a registered processor class by its ``type`` tag.

        Raises:
            KeyError: If the type name is not found in the registry.
        """
        if type_name not in cls._registry:
            available = sorted(cls._registry)
            msg = f"Unknown processor type: {type_name!r}. Choose from: {available}"
            raise KeyError(msg)
        return cls._registry[type_name]

    @classmethod
    def from_config(cls: type[ProcessorRegistry], step: dict[str, Any]) -> BaseProcessor:
        """Reconstruct a single processor from a serialized config dict."""
        return cls.get(step["type"]).model_validate(step)


@ProcessorRegistry.register
class NormalizationProcessor(BaseProcessor):
    """Normalizes ODMR data by dividing each pixel's spectrum by a scalar factor.

    ``method='mean'`` is the physically correct choice: mean-normalization
    preserves per-pixel off-resonance baseline variation, which is required for
    downstream fluorescence correction.

    ``method='max'`` is **deprecated** and will be removed in a future release.
    It forces every pixel's off-resonance level to exactly 1.0, destroying the
    baseline information needed for fluorescence correction. It is retained only
    to allow direct comparison with older pipelines during investigation.

    Attributes:
        method: The normalization method. ``'mean'`` (default) or ``'max'`` (deprecated).
    """

    METHODS: ClassVar[tuple[str, ...]] = ("mean", "max")  # "mean" is default; "max" is deprecated

    type: Literal["NormalizationProcessor"] = "NormalizationProcessor"
    method: Literal["max", "mean"] = "mean"

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Warn for max-normalization; reject unknown methods at construction time."""
        import warnings

        if v == "max":
            warnings.warn(
                "method='max' is deprecated and will be removed in a future release. "
                "Max-normalization forces every pixel's off-resonance level to exactly 1.0, "
                "destroying the per-pixel baseline variation required for fluorescence "
                "correction. Use method='mean' for physically correct results.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif v != "mean":
            raise ValueError(
                f"Unsupported normalization method '{v}'. Use 'mean' or 'max' (deprecated)."
            )
        return v

    def process(self, data: ODMRData) -> ODMRData:
        """Normalize the data per pixel across the frequency dimension."""
        from qdmpy.odmr.data import ODMRData

        logger.debug("Normalizing data using method: {}", self.method)
        if self.method == "max":
            factors = data.data.max(dim="freq_idx")
        else:
            factors = data.data.mean(dim="freq_idx")

        n_zero = int((factors == 0).sum())
        if n_zero:
            # An unguarded zero factor divides to NaN (equal values) or +-inf
            # (values that cancel to a zero mean) with no diagnostic -- force
            # NaN explicitly and warn, rather than let inf leak downstream.
            logger.warning(
                "NormalizationProcessor: {} pixel(s) have a zero {} factor and will be NaN",
                n_zero,
                self.method,
            )
        factors = factors.where(factors != 0)

        normalized = data.data / factors
        return ODMRData(data=normalized, metadata=data.metadata.copy())


@ProcessorRegistry.register
class BinningProcessor(BaseProcessor):
    """Spatial binning of ODMR data using xarray coarsen.

    Attributes:
        bin_factor: The factor by which to bin the data spatially.
    """

    type: Literal["BinningProcessor"] = "BinningProcessor"
    bin_factor: int = Field(gt=0)

    def process(self, data: ODMRData) -> ODMRData:
        """Bin the data spatially by the specified factor."""
        from qdmpy.exceptions import DataShapeError
        from qdmpy.odmr.data import ODMRData

        # Skip binning if factor is 1 (no binning)
        if self.bin_factor == 1:
            logger.debug("Bin factor is 1, skipping binning")
            return data

        n_y, n_x = data.data.sizes["y"], data.data.sizes["x"]
        if self.bin_factor > min(n_y, n_x):
            # coarsen(..., boundary="trim") silently trims to a zero-sized
            # array rather than raising when bin_factor exceeds the scan
            # dimensions -- fail loudly instead of producing an empty map.
            msg = (
                f"bin_factor={self.bin_factor} exceeds the scan dimensions "
                f"({n_y}, {n_x}); binning would produce an empty array"
            )
            raise DataShapeError(msg)

        logger.debug("Binning data with factor: {}", self.bin_factor)
        binned = data.data.coarsen(y=self.bin_factor, x=self.bin_factor, boundary="trim").mean()  # type: ignore[attr-defined]
        return ODMRData(data=binned, metadata=data.metadata.copy())


@ProcessorRegistry.register
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

        logger.debug("Masking outliers with z_score_threshold: {}", self.z_score_threshold)
        data_mean = data.data.mean(dim="freq_idx")
        data_std = data.data.std(dim="freq_idx")
        z_scores = np.abs((data.data - data_mean) / (data_std + 1e-10))
        mask = z_scores > self.z_score_threshold
        processed = data.data.where(~mask)
        return ODMRData(data=processed, metadata=data.metadata.copy())


@ProcessorRegistry.register
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

        logger.info("Applying fluorescence correction with factor: {}", self.correction_factor)
        _, baseline_corrected = analyze_fluorescence_effects(data)
        correction = self.correction_factor * baseline_corrected
        processed = data.data - correction
        return ODMRData(data=processed, metadata=data.metadata.copy())


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
            logger.info("Automatically selected pixel index: {}", flat_idx)
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
        logger.debug("Adding processor: {}", processor.__class__.__name__)
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
            logger.debug("Applying processor: {}", processor.__class__.__name__)
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
            manager.add_processor(ProcessorRegistry.from_config(step))
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
