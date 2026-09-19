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
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.ndimage import median_filter

from qdmpy.exceptions import DataShapeError, DataValidationError, ParameterError

# A background-fit mask is a (row_indices, col_indices) pair.
_MASK_TUPLE_LEN = 2

# Hot-pixel detection: neighbourhood for the local median, and the factor
# converting a median absolute deviation to a Gaussian-equivalent sigma.
_LOCAL_MEDIAN_SIZE = 3
_MAD_TO_SIGMA = 1.4826


class BaseFieldProcessor(BaseModel):
    """Abstract base for all field-map processors.

    Processors are Pydantic frozen models: all configuration lives in fields
    set at construction; ``process()`` is a pure function of its argument.
    ``pixel_spacing`` (in metres) must be present in ``field_map.attrs``.

    ``extra='forbid'``: an unknown keyword is a typo, and silently dropping
    it means the processor runs with a default the caller did not intend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
            DataValidationError: If pixel_spacing not in attrs.
        """
        if "pixel_spacing" not in field_map.attrs:
            msg = "field_map.attrs must contain 'pixel_spacing' (metres)"
            raise DataValidationError(msg)
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
        logger.info("Running field processing pipeline ({} steps)", len(self._processors))
        result = field_map.copy(deep=True)
        for proc in self._processors:
            result = proc.process(result)
            logger.debug("Applied {}: output shape {}", proc.__class__.__name__, result.shape)
        return result


class HotPixelFilter(BaseFieldProcessor):
    """Detect and replace isolated outlier pixels in a field map.

    A pixel is flagged when it departs from the median of its 3x3
    neighbourhood by more than ``threshold_sigma`` robust standard deviations
    of that local residual (sigma = 1.4826 * MAD). An optional absolute
    threshold pre-filter flags ``|field| > absolute_threshold`` as well.

    Why a *local* residual: a magnetic field measured at standoff h is smooth
    over ~h, while a hot pixel differs from its immediate neighbours. Both
    global statistics fail on real maps (mostly flat, sparse strong features).
    On a synthetic map with noise, a dipole and 20 injected 2 uT spikes:

    - global std (the previous detector) caught 0/20 spikes and still
      flagged 49 of 337 real-feature pixels -- the features inflate the std;
    - global MAD caught 20/20 spikes but flagged all 337 feature pixels;
    - the local 3x3 residual caught 20/20 spikes and flagged 30 feature
      pixels, all at the dipole core where the feature is only ~4 px wide.

    Replacement uses the original map, never values replaced earlier in the
    same pass, and excludes other flagged pixels from the neighbour mean.
    """

    threshold_sigma: float = Field(
        default=5.0, gt=0, description="Robust-sigma threshold on the local residual"
    )
    window_size: int = Field(default=3, ge=1, description="Half-width of replacement window")
    replacement: Literal["mean", "nan", "zero"] = Field(
        default="mean", description="Replacement strategy"
    )
    absolute_threshold: float | None = Field(
        default=None, description="Absolute threshold: filter |field| > this first"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Detect and replace outlier pixels.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            New DataArray with outliers replaced.
        """
        _ = self._pixel_spacing(field_map)  # Validate
        logger.info(
            "Detecting hot pixels (threshold_sigma={}, replacement={})",
            self.threshold_sigma,
            self.replacement,
        )
        original = field_map.values
        outlier_mask = self._detect(original)
        logger.info("Flagged {} hot pixel(s)", int(outlier_mask.sum()))
        data = self._replace(original, outlier_mask)
        return xr.DataArray(
            data, dims=field_map.dims, coords=field_map.coords, attrs=field_map.attrs
        )

    def _detect(self, data: NDArray) -> NDArray:
        """Boolean mask of pixels that stand out from their 3x3 neighbourhood."""
        finite = np.isfinite(data)
        filled = np.where(finite, data, np.nanmedian(data))
        residual = filled - median_filter(filled, size=_LOCAL_MEDIAN_SIZE, mode="reflect")
        centred = residual[finite] - np.median(residual[finite])
        sigma = _MAD_TO_SIGMA * float(np.median(np.abs(centred))) if centred.size else 0.0
        mask = (
            finite & (np.abs(residual) > self.threshold_sigma * sigma)
            if sigma > 0
            else (np.zeros_like(finite))
        )
        if self.absolute_threshold is not None:
            mask |= finite & (np.abs(data) > self.absolute_threshold)
        return mask

    def _replace(self, original: NDArray, outlier_mask: NDArray) -> NDArray:
        """Replace flagged pixels; windows always read the unmodified input."""
        data = original.copy()
        if self.replacement == "nan":
            data[outlier_mask] = np.nan
            return data
        if self.replacement == "zero":
            data[outlier_mask] = 0.0
            return data

        good = np.where(outlier_mask, np.nan, original)  # never average other outliers
        h, w = original.shape
        k = self.window_size
        for r, c in np.argwhere(outlier_mask):
            window = good[max(0, r - k) : min(h, r + k + 1), max(0, c - k) : min(w, c + k + 1)]
            finite = window[np.isfinite(window)]
            data[r, c] = float(finite.mean()) if finite.size else np.nan
        return data


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

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("mask")
    @classmethod
    def validate_mask(
        cls, v: tuple[tuple[int, ...], ...] | None
    ) -> tuple[tuple[int, ...], ...] | None:
        """Require exactly two index sequences of equal length.

        The field type permits any number of sequences, but ``process()``
        unpacks it as ``(row_indices, col_indices)`` -- anything else raised a
        bare unpack error deep inside the fit.
        """
        if v is None:
            return v
        if len(v) != _MASK_TUPLE_LEN:
            msg = f"mask must be a (row_indices, col_indices) pair, got {len(v)} sequence(s)"
            raise ValueError(msg)
        if len(v[0]) != len(v[1]):
            msg = (
                f"mask row_indices and col_indices must be the same length, "
                f"got {len(v[0])} and {len(v[1])}"
            )
            raise ValueError(msg)
        return v

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Remove polynomial background.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            New DataArray with background subtracted.
        """
        _ = self._pixel_spacing(field_map)  # Validate
        logger.info("Removing polynomial background (degree={})", self.degree)
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
            msg = f"degree must be 0, 1, or 2; got {self.degree}"
            raise ParameterError(msg)

        # Determine which pixels to use for fit. Non-finite samples are
        # excluded: lstsq propagates a single NaN into every coefficient, so
        # one dead pixel used to NaN the entire background-subtracted map.
        flat = data.ravel()
        active = np.isfinite(flat)
        n_nonfinite = int((~active).sum())
        if n_nonfinite:
            logger.warning(
                "Excluding {} non-finite pixel(s) from the degree-{} background fit",
                n_nonfinite,
                self.degree,
            )

        if self.mask is not None:
            mask_rows, mask_cols = self.mask
            for r, c in zip(mask_rows, mask_cols, strict=True):
                if 0 <= r < h and 0 <= c < w:
                    active[r * w + c] = False

        n_coeffs = features.shape[1]
        if int(active.sum()) < n_coeffs:
            msg = (
                f"Only {int(active.sum())} usable pixel(s) remain after excluding "
                f"non-finite and masked values; a degree-{self.degree} fit needs at "
                f"least {n_coeffs}"
            )
            raise DataShapeError(msg)

        # Fit: lstsq on active pixels only
        coeffs = np.linalg.lstsq(features[active], flat[active], rcond=None)[0]

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
    padding_factor: float = Field(
        default=3.0, ge=1.0, description="Padded size as a multiple of the map size (>= 1)"
    )
    oversampling: int = Field(default=2, description="FFT oversampling factor")

    model_config = ConfigDict(frozen=True, extra="forbid")

    def process(self, field_map: xr.DataArray) -> xr.DataArray:
        """Apply upward/downward continuation.

        Args:
            field_map: DataArray with pixel_spacing in attrs.

        Returns:
            New DataArray with continuation applied.
        """
        ps = self._pixel_spacing(field_map)
        logger.info("Applying upward/downward continuation (dz={} m)", self.dz)
        data = field_map.values
        h, w = data.shape

        if self.dz == 0:
            return field_map.copy(deep=True)

        if self.dz < 0:
            logger.warning(
                "Downward continuation (dz={} m < 0) amplifies high frequencies and noise",
                self.dz,
            )

        # Remove the mean before zero-padding and restore it afterwards. The
        # mean is the k=0 component, which continuation leaves unchanged
        # (H(0) = 1), so this is exact -- but left in, the zero pad is a step
        # of size mean(data) that rings back into the map. Validated on an
        # analytic dipole with a 5 uT offset: RMS error 0.71 -> 0.0013 uT.
        mean = float(np.nanmean(data))

        # Pad
        pad_h = int(h * self.padding_factor)
        pad_w = int(w * self.padding_factor)
        padded = np.zeros((pad_h, pad_w))
        offset_h = (pad_h - h) // 2
        offset_w = (pad_w - w) // 2
        padded[offset_h : offset_h + h, offset_w : offset_w + w] = data - mean

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
        result = out_padded[offset_h : offset_h + h, offset_w : offset_w + w] + mean

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

    model_config = ConfigDict(frozen=True, extra="forbid")

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
        logger.info("Subtracting blank map")
        blank_array = np.array(self.blank)

        if blank_array.shape != field_map.shape:
            raise DataShapeError(
                f"Blank shape {blank_array.shape} != field shape {field_map.shape}"
            )

        result = field_map.values - blank_array

        return xr.DataArray(
            result, dims=field_map.dims, coords=field_map.coords, attrs=field_map.attrs
        )
