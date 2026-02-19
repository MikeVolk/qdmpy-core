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

from QDMpy.constants import DEFAULT_VMAX, DEFAULT_VMIN, PROMINENCE
from QDMpy.exceptions import (
    DataShapeError,
    DataValidationError,
    ModelGuessNotPossibleError,
    ModelNotFoundError,
    ParameterError,
)
from QDMpy.fitting.models import ModelRegistry

if TYPE_CHECKING:
    from QDMpy.fitting.models import Model


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


def guess_initial_fit_parameters(data: NDArray, freq: NDArray, model: Model) -> NDArray:
    """Guess initial fit parameters based on the selected model.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: Frequency array.
        model: Model instance.

    Returns:
        Initial parameters array (n_pol, n_frange, n_pixel, n_params).
    """
    from QDMpy.exceptions import ParameterError

    parameter_guessers = {
        "center": lambda: guess_center(data, freq),
        "contrast": lambda: guess_contrast(data),
        "width": lambda: guess_width(data, freq, DEFAULT_VMIN, DEFAULT_VMAX),
        "offset": lambda: np.ones((data.shape[0], data.shape[1], data.shape[2])),
    }

    fit_parameters = []
    for param in model.parameter_names:
        param_type = model.parameter_types[param]
        if param_type in parameter_guessers:
            fit_parameters.append(parameter_guessers[param_type]())
        else:
            msg = f"Parameter '{param}' has no defined guess method."
            raise ParameterError(msg)

    return np.stack(fit_parameters, axis=-1)


@njit(parallel=True, fastmath=True)
def guess_contrast(data: NDArray) -> NDArray:  # pragma: no cover
    """Estimate contrast for each pixel.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape
    amp = np.zeros((n_pol, n_frange, n_pixel))
    for p in range(n_pol):
        for r in range(n_frange):
            for px in prange(n_pixel):  # type: ignore[not-iterable]
                amp[p, r, px] = guess_contrast_pixel(data[p, r, px, :])
    return amp


@njit(fastmath=True)
def guess_contrast_pixel(pixel: NDArray) -> float:  # pragma: no cover
    """Estimate contrast for a single pixel's frequency spectrum."""
    mx, mn = np.nanmax(pixel), np.nanmin(pixel)
    return 0 if mx == 0 else abs((mx - mn) / mx)


@njit(parallel=True, fastmath=True)
def guess_center(data: NDArray, freq: NDArray) -> NDArray:  # pragma: no cover
    """Guess the center frequency for each pixel.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: Frequency array.

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape
    centers = np.zeros((n_pol, n_frange, n_pixel))
    for p in range(n_pol):
        for r in range(n_frange):
            for px in prange(n_pixel):  # type: ignore[not-iterable]
                centers[p, r, px] = guess_center_pixel(data[p, r, px, :], freq[r])
    return centers


@njit(fastmath=True)
def guess_center_pixel(pixel: NDArray, freq: NDArray) -> float:  # pragma: no cover
    """Guess center frequency of a single pixel."""
    normalized = normalize_pixel(pixel)
    idx = np.argmin(np.abs(normalized - 0.5))
    return freq[idx]


@njit(parallel=True, fastmath=True)
def guess_width(data: NDArray, freq: NDArray, vmin: float, vmax: float) -> NDArray:  # pragma: no cover
    """Guess width of ODMR resonance peaks.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: Frequency array.
        vmin: Min normalized cumsum value.
        vmax: Max normalized cumsum value.

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape
    widths = np.zeros((n_pol, n_frange, n_pixel))
    for p in range(n_pol):
        for r in range(n_frange):
            for px in prange(n_pixel):  # type: ignore[not-iterable]
                widths[p, r, px] = guess_width_pixel(
                    data[p, r, px, :],
                    freq[r],
                    vmin,
                    vmax,
                )
    return widths


@njit(fastmath=True)
def guess_width_pixel(pixel: NDArray, freq: NDArray, vmin: float, vmax: float) -> float:  # pragma: no cover
    """Guess width of a single pixel's resonance."""
    normalized = normalize_pixel(pixel)
    lidx = np.argmin(np.abs(normalized - vmin))
    ridx = np.argmin(np.abs(normalized - vmax))
    return abs(freq[ridx] - freq[lidx])
