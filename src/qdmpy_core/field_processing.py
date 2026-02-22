"""Field map processing pipeline for post-fit magnetic field data.

Provides composable, immutable processors for preprocessing B111 maps and other
field data. All processors are frozen Pydantic models operating on xarray DataArrays.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import xarray as xr
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from qdmpy_core.exceptions import DataShapeError


class BaseFieldProcessor(BaseModel):
    """Abstract base for all field-map processors.

    Processors are Pydantic frozen models: all configuration lives in fields
    set at construction; ``process()`` is a pure function of its argument.
    ``pixel_spacing`` (in metres) must be present in ``field_map.attrs``.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Transform a (H, W) field map.

        Args:
            field_map: DataArray with dims (y, x), values in µT,
                       and ``pixel_spacing`` (float, metres) in ``.attrs``.

        Returns:
            Processed DataArray with identical dims, coords, and attrs.
            Input is never mutated.
        """

    @staticmethod
    def _pixel_spacing(field_map: xr.DataArray) -> float:
        """Extract pixel spacing from field map attributes.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            Pixel spacing in metres.

        Raises:
            ValueError: If pixel_spacing not in attrs.
        """
        if "pixel_spacing" not in field_map.attrs:
            raise ValueError("field_map.attrs must contain 'pixel_spacing' (metres)")
        return float(field_map.attrs["pixel_spacing"])


class FieldProcessingPipeline:
    """Sequential chain of BaseFieldProcessor steps operating on xr.DataArray.

    Processors are applied in order; each receives output of the previous.
    Implements fluent API for method chaining.
    """

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self._processors: list[BaseFieldProcessor] = []

    def add(self, processor: BaseFieldProcessor) -> FieldProcessingPipeline:
        """Append a processor.

        Args:
            processor: A BaseFieldProcessor subclass instance.

        Returns:
            self for method chaining.
        """
        self._processors.append(processor)
        return self

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Apply all processors in order.

        Args:
            field_map: Input DataArray. Never mutated.

        Returns:
            Processed DataArray with same dims, coords, attrs.
            If pipeline is empty, returns a deep copy of input.
        """
        result = field_map.copy(deep=True)
        for proc in self._processors:
            result = proc.process(result)
            logger.debug(
                "Field processor applied",
                processor=proc.__class__.__name__,
                shape=result.shape,
            )
        return result


class HotPixelFilter(BaseFieldProcessor):
    """Detect and replace outlier pixels in a field map.

    Uses median ± sigma threshold with optional absolute threshold pre-filter.
    Replacement can be mean of neighbors, NaN, or zero.
    """

    threshold_sigma: float = Field(default=5.0, description="Sigma threshold for outlier detection")
    window_size: int = Field(default=3, description="Half-width of replacement window")
    replacement: Literal["mean", "nan", "zero"] = Field(
        default="mean", description="Replacement strategy"
    )
    absolute_threshold: float | None = Field(
        default=None, description="Absolute threshold: filter |field| > this first"
    )

    model_config = ConfigDict(frozen=True)

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Detect and replace outlier pixels.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            New DataArray with outliers replaced.
        """
        _ = self._pixel_spacing(field_map)  # Validate
        data = field_map.values.copy()

        # Compute median and std
        median = np.nanmedian(data)
        std = np.nanstd(data)

        # Outlier mask: |value - median| > threshold_sigma * std
        outlier_mask = np.abs(data - median) > self.threshold_sigma * std

        # Pre-filter: absolute threshold
        if self.absolute_threshold is not None:
            abs_mask = np.abs(data) > self.absolute_threshold
            outlier_mask = outlier_mask | abs_mask

        # Replace outliers
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                if outlier_mask[r, c]:
                    # Extract window (clip at boundaries)
                    r_min = max(0, r - self.window_size)
                    r_max = min(data.shape[0], r + self.window_size + 1)
                    c_min = max(0, c - self.window_size)
                    c_max = min(data.shape[1], c + self.window_size + 1)

                    window = data[r_min:r_max, c_min:c_max].copy()
                    # Exclude center pixel
                    window[r - r_min, c - c_min] = np.nan

                    if self.replacement == "mean":
                        data[r, c] = np.nanmean(window)
                    elif self.replacement == "nan":
                        data[r, c] = np.nan
                    elif self.replacement == "zero":
                        data[r, c] = 0.0

        return xr.DataArray(
            data, dims=field_map.dims, coords=field_map.coords, attrs=field_map.attrs
        )


class QuadraticBackgroundSubtractor(BaseFieldProcessor):
    """Remove polynomial background via least-squares fit.

    Fits a polynomial surface of given degree to the field, using optional mask
    to exclude pixels. Subtracts the fitted surface from all pixels.
    """

    degree: int = Field(default=2, description="Polynomial degree (0=const, 1=plane, 2=quadratic)")
    mask: tuple[tuple[int, ...], ...] | None = Field(
        default=None,
        description="Tuple of (row_indices, col_indices) to EXCLUDE from fit",
    )

    model_config = ConfigDict(frozen=True)

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Remove polynomial background.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            New DataArray with background subtracted.
        """
        _ = self._pixel_spacing(field_map)  # Validate
        data = field_map.values
        h, w = data.shape

        # Build feature matrix (all pixels)
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)

        # Normalize x, y to [-1, 1] for numerical stability
        x_norm = 2 * X / (w - 1) - 1 if w > 1 else np.zeros_like(X)
        y_norm = 2 * Y / (h - 1) - 1 if h > 1 else np.zeros_like(Y)

        # Build polynomial features
        if self.degree == 0:
            features = np.ones((h * w, 1))
        elif self.degree == 1:
            features = np.column_stack(
                [
                    np.ones(h * w),
                    x_norm.ravel(),
                    y_norm.ravel(),
                ]
            )
        elif self.degree == 2:
            features = np.column_stack(
                [
                    np.ones(h * w),
                    x_norm.ravel(),
                    y_norm.ravel(),
                    x_norm.ravel() ** 2,
                    (x_norm * y_norm).ravel(),
                    y_norm.ravel() ** 2,
                ]
            )
        else:
            raise ValueError(f"degree must be 0, 1, or 2; got {self.degree}")

        # Determine which pixels to use for fit
        if self.mask is None:
            active = np.ones(h * w, dtype=bool)
        else:
            active = np.ones(h * w, dtype=bool)
            mask_rows, mask_cols = self.mask
            for r, c in zip(mask_rows, mask_cols, strict=True):
                if 0 <= r < h and 0 <= c < w:
                    active[r * w + c] = False

        # Fit: lstsq on active pixels only
        coeffs = np.linalg.lstsq(features[active], data.ravel()[active], rcond=None)[0]

        # Evaluate surface at all pixels
        surface = features @ coeffs
        surface = surface.reshape(h, w)

        # Subtract
        result = data - surface

        return xr.DataArray(
            result, dims=field_map.dims, coords=field_map.coords, attrs=field_map.attrs
        )


class UpwardContinuation(BaseFieldProcessor):
    """Upward/downward continuation in Fourier space.

    Applies frequency-domain filter to attenuate high-wavenumber components,
    simulating field at a different height above the source.
    """

    dz: float = Field(description="Continuation height in metres (>0=up, <0=down)")
    padding_factor: float = Field(default=3.0, description="Padding multiplier")
    oversampling: int = Field(default=2, description="FFT oversampling factor")

    model_config = ConfigDict(frozen=True)

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Apply upward/downward continuation.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            New DataArray with continuation applied.
        """
        ps = self._pixel_spacing(field_map)
        data = field_map.values
        h, w = data.shape

        if self.dz == 0:
            return field_map.copy(deep=True)

        if self.dz < 0:
            logger.warning(
                "Downward continuation (dz < 0) amplifies high frequencies and noise",
                dz=self.dz,
            )

        # Pad
        pad_h = int(h * self.padding_factor)
        pad_w = int(w * self.padding_factor)
        padded = np.zeros((pad_h, pad_w))
        offset_h = (pad_h - h) // 2
        offset_w = (pad_w - w) // 2
        padded[offset_h : offset_h + h, offset_w : offset_w + w] = data

        # Oversampled FFT
        fft_h = pad_h * self.oversampling
        fft_w = pad_w * self.oversampling

        # Wavenumber grid
        fy = np.fft.fftfreq(fft_h, d=ps)
        fx = np.fft.fftfreq(fft_w, d=ps)
        Fx, Fy = np.meshgrid(fx, fy)
        k = 2 * np.pi * np.sqrt(Fx**2 + Fy**2)

        # Continuation filter
        H = np.exp(-self.dz * k)

        # Apply
        F = np.fft.fft2(padded, s=(fft_h, fft_w))
        F_cont = F * H
        out = np.real(np.fft.ifft2(F_cont))

        # Crop back: first to padded size, then to original
        out_padded = out[:pad_h, :pad_w]
        result = out_padded[offset_h : offset_h + h, offset_w : offset_w + w]

        return xr.DataArray(
            result, dims=field_map.dims, coords=field_map.coords, attrs=field_map.attrs
        )


class BlankSubtractor(BaseFieldProcessor):
    """Subtract a pre-measured blank map (background).

    The blank must have the same shape as the field map.
    """

    blank: tuple[tuple[float, ...], ...] = Field(
        description="Blank map as nested tuple (must match field shape)"
    )

    model_config = ConfigDict(frozen=True)

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Subtract blank from field.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            New DataArray with blank subtracted.

        Raises:
            DataShapeError: If blank shape != field shape.
        """
        _ = self._pixel_spacing(field_map)  # Validate
        blank_array = np.array(self.blank)

        if blank_array.shape != field_map.shape:
            raise DataShapeError(
                f"Blank shape {blank_array.shape} != field shape {field_map.shape}"
            )

        result = field_map.values - blank_array

        return xr.DataArray(
            result, dims=field_map.dims, coords=field_map.coords, attrs=field_map.attrs
        )
