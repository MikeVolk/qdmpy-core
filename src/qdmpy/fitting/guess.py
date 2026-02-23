"""Module for guessing models and initial fit parameters for ODMR data.

The numba-jitted functions in this module operate on 4D numpy arrays with the
convention: (n_polarity, n_freq_range, n_pixel, n_frequency).

Higher-level functions that accept xr.DataArray (or ODMRData) extract numpy
arrays at the boundary before calling into numba.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numba import njit, prange
from numpy.typing import NDArray
from scipy.signal import find_peaks

from qdmpy.constants import PROMINENCE
from qdmpy.exceptions import (
    DataShapeError,
    DataValidationError,
    ModelGuessNotPossibleError,
    ModelNotFoundError,
)
from qdmpy.fitting.models import ModelRegistry

if TYPE_CHECKING:
    from qdmpy.fitting.models import Model


@njit(fastmath=True)
def normalize_pixel(pixel: NDArray) -> NDArray:  # pragma: no cover
    """Normalize a pixel's cumulative sum.

    Args:
        pixel: 1D array of intensity values for a single pixel.

    Returns:
        The normalized cumulative sum of the pixel data.
    """
    pixel = np.cumsum(pixel - 1)
    pixel -= np.min(pixel)
    max_val = np.max(pixel)
    return pixel / max_val if max_val > 0 else pixel


def validate_array(data: NDArray, expected_dim: int, name: str) -> None:
    """Validate that an array has the expected number of dimensions."""
    if data is None:
        msg = f"{name} cannot be None."
        raise DataValidationError(msg)
    if not np.issubdtype(data.dtype, np.number):
        msg = f"{name} must be a numeric array."
        raise DataValidationError(msg)
    if data.ndim != expected_dim:
        msg = f"{name} must have {expected_dim} dimensions. Got {data.ndim}."
        raise DataShapeError(msg)


def guess_model(data: NDArray) -> Model:
    """Automatically determine the best fitting model for ODMR data.

    Args:
        data: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        An instance of the appropriate model.

    Raises:
        ModelGuessNotPossibleError: If the model cannot be reliably determined.
    """
    logger.info("Trying to detect best fitting model for ODMR data.")
    n_peaks, doubt, _ = guess_n_peaks(data)

    if not doubt:
        model = get_model_by_peaks(n_peaks)
        logger.info(f"Detected model: {model.name}")
        return model
    msg = "Guessing the model is not possible. Please select model manually."
    raise ModelGuessNotPossibleError(msg)


def guess_n_peaks(data: NDArray) -> tuple[int, bool, list[NDArray]]:
    """Estimate the number of peaks in ODMR data.

    Args:
        data: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        Tuple of (n_peaks, doubt, peak_indices_list).
    """
    validate_array(data, 4, "data")
    # Median across pixels (axis 2) gives (n_pol, n_frange, n_freq)
    median_data = np.median(data, axis=2)
    indices = [
        find_peaks(-median_data[p, f], prominence=PROMINENCE)[0]
        for p, f in np.ndindex(*data.shape[:2])
    ]
    n_peaks = int(np.round(np.mean([len(idx) for idx in indices])))
    doubt = np.std([len(idx) for idx in indices]) != 0
    return n_peaks, doubt, indices


def get_model_by_peaks(n_peaks: int) -> Model:
    """Retrieve the model instance based on the number of peaks."""
    for model_cls in ModelRegistry.all().values():
        model_instance = model_cls()  # type: ignore[call-arg]
        if model_instance.n_peaks == n_peaks:
            return model_instance
    msg = f"No model found for {n_peaks} peaks."
    raise ModelNotFoundError(msg)


@njit(parallel=True, fastmath=True)
def cumsum_contrast(data: NDArray) -> NDArray:  # pragma: no cover
    """Estimate contrast for each pixel using a single flat parallel loop.

    Flattens (n_pol, n_frange, n_pixel) into one prange, exposing all pixels
    across all polarities and frequency ranges to the thread pool simultaneously.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, _ = data.shape
    total = n_pol * n_frange * n_pixel
    amp = np.zeros((n_pol, n_frange, n_pixel))
    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)
        mx = np.nanmax(data[p, r, px])
        mn = np.nanmin(data[p, r, px])
        amp[p, r, px] = 0.0 if mx == 0.0 else abs((mx - mn) / mx)
    return amp


@njit(parallel=True, fastmath=True)
def cumsum_center(data: NDArray, freq: NDArray) -> NDArray:  # pragma: no cover
    """Guess the center frequency for each pixel using a single flat parallel loop.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: 2D frequency array (n_frange, n_freq).

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, _ = data.shape
    total = n_pol * n_frange * n_pixel
    centers = np.zeros((n_pol, n_frange, n_pixel))
    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)
        norm = normalize_pixel(data[p, r, px])
        centers[p, r, px] = freq[r, np.argmin(np.abs(norm - 0.5))]
    return centers


@njit(parallel=True, fastmath=True)
def cumsum_width(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> NDArray:  # pragma: no cover
    """Guess width of ODMR resonance peaks using a single flat parallel loop.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: 2D frequency array (n_frange, n_freq).
        vmin: Min normalized cumsum threshold.
        vmax: Max normalized cumsum threshold.

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, _ = data.shape
    total = n_pol * n_frange * n_pixel
    widths = np.zeros((n_pol, n_frange, n_pixel))
    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)
        norm = normalize_pixel(data[p, r, px])
        lidx = np.argmin(np.abs(norm - vmin))
        ridx = np.argmin(np.abs(norm - vmax))
        widths[p, r, px] = abs(freq[r, ridx] - freq[r, lidx])
    return widths
