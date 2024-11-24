import numpy as np
import numba
from numba import njit, prange
from numpy.typing import NDArray

from __future__ import annotations
import os
import sys
import logging
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np
from scipy.signal import find_peaks

LOG = logging.getLogger(__name__)

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)
import QDMpy
from QDMpy._core.models import IMPLEMENTED

def guess_model_name(data) -> str:
    """Guess the model name from the data.

    Returns:
        str: Name of the model.
    """

    data = np.median(data, axis=3)
    n_peaks, doubt, _ = guess_model(data)

    if doubt:
        LOG.warning(
            "Doubt on the diamond type. Check using `guess_diamond_type('debug')` and set manually if incorrect."
        )

    model = [mdict for mdict in IMPLEMENTED.values() if mdict["n_peaks"] == n_peaks][0]

    LOG.info(
        f"Guessed diamond type: {n_peaks} peaks -> {model['func_name']} ({model['name']})"
    )
    return model["func_name"]


def guess_model(data: NDArray) -> tuple[int, bool, Any]:
    """Guess the diamond type based on the number of peaks.

    :return: diamond_type (int)

    Args:
        data (np.ndarray): data


    Returns:

    """

    indices = []

    # Find the indices of the peaks
    for p, f in np.ndindex(*data.shape[:2]):
        peaks = find_peaks(
            -data[p, f], prominence=QDMpy.SETTINGS["model"]["find_peaks"]["prominence"]
        )
        indices.append(peaks[0])

    n_peaks = int(np.round(np.mean([len(idx) for idx in indices])))

    doubt = np.std([len(idx) for idx in indices]) != 0

    return n_peaks, doubt, peaks

def guess_initial_fit_parameters(
    data: NDArray, freq: NDArray, model: Dict[str, Any]
) -> NDArray:
    """
    Guess initial fit parameters based on the selected model.

    :param data: NDArray
        3D array of the data to fit (e.g., ODMR data).
    :param freq: NDArray
        1D array of the frequencies corresponding to the data.
    :param model: Dict[str, Any]
        Model dictionary containing parameter names and number of peaks.

    :return: NDArray
        Initial fit parameters as a 4D array of shape (n_pol, n_frange, n_pixel, n_params).
    """
    # Define parameter guessers for each parameter type
    parameter_guessers = {
        "center": lambda: guess_center(data, freq),
        "contrast": lambda: guess_contrast(data),
        "width": lambda: guess_width(
            data, freq, vmin=0.3, vmax=0.7
        ),  # Default vmin/vmax can be adjusted dynamically
        "offset": lambda: np.ones(data.shape[:-1]),  # Offset is often assumed to be 1
    }

    # Initialize list for parameter arrays
    fit_parameters = []

    # Guess each parameter defined in the model
    for param in model["params"]:
        if param in parameter_guessers:
            fit_parameters.append(parameter_guessers[param]())
        else:
            raise ValueError(f"Parameter {param} has no defined guess method.")

    # Stack parameters along the last axis
    return np.stack(fit_parameters, axis=-1)


@njit(parallel=True)
def guess_contrast(data: NDArray) -> NDArray:
    """
    Guess the contrast of ODMR data.

    :param data: np.array
        Data to guess the contrast from
    :return: np.array
        Contrast of the data
    """
    amp = np.zeros(data.shape[:-1])  # Match shape correctly
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            for p in range(data.shape[2]):
                amp[i, j, p] = guess_contrast_pixel(data[i, j, p])
    return amp


@njit
def guess_contrast_pixel(pixel: NDArray) -> float:
    """
    Guess the contrast of a single pixel.

    :param pixel: np.array
        Pixel data
    :return: float
        Contrast of the pixel
    """
    mx = np.nanmax(pixel)
    mn = np.nanmin(pixel)
    if mx == 0:
        return 0  # Avoid division by zero
    return np.abs((mx - mn) / mx)


@njit(parallel=True, fastmath=True)
def guess_center(data: NDArray, freq: NDArray) -> NDArray:
    """
    Guess the center frequency of ODMR data.

    :param data: np.array
        Data to guess the center frequency from
    :param freq: np.array
        Frequency range of the data
    :return: np.array
        Center frequency of the data
    """
    center = np.zeros(data.shape[:-1])
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            for p in range(data.shape[2]):
                center[i, j, p] = guess_center_pixel(data[i, j, p], freq)
    return center


@njit(fastmath=True)
def guess_center_pixel(pixel: NDArray, freq: NDArray) -> float:
    """
    Guess the center frequency of a single pixel.

    :param pixel: np.array
        Pixel data
    :param freq: np.array
        Frequency range of the data
    :return: float
        Center frequency of the pixel
    """
    pixel = normalized_cumsum_pixel(pixel)
    idx = np.argmin(np.abs(pixel - 0.5))
    return freq[idx]


@njit(parallel=True, fastmath=True)
def guess_width(data: NDArray, freq: NDArray, vmin: float, vmax: float) -> NDArray:
    """
    Guess the width of ODMR resonance peaks.

    :param data: np.array
        Data to guess the width from
    :param freq: np.array
        Frequency range of the data
    :param vmin: float
        Minimum value of normalized cumsum to be considered
    :param vmax: float
        Maximum value of normalized cumsum to be considered
    :return: np.array
        Width of the data
    """
    width = np.zeros(data.shape[:-1])
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            for p in range(data.shape[2]):
                width[i, j, p] = guess_width_pixel(data[i, j, p], freq, vmin, vmax)
    return width


@njit(fastmath=True)
def guess_width_pixel(pixel: NDArray, freq: NDArray, vmin: float, vmax: float) -> float:
    """
    Guess the width of a single pixel.

    :param pixel: np.array
        Pixel data
    :param freq: np.array
        Frequency range of the data
    :param vmin: float
        Minimum value of normalized cumsum to be considered
    :param vmax: float
        Maximum value of normalized cumsum to be considered
    :return: float
        Width of the pixel
    """
    pixel = normalized_cumsum_pixel(pixel)
    lidx = np.argmin(np.abs(pixel - vmin))
    ridx = np.argmin(np.abs(pixel - vmax))
    return freq[ridx] - freq[lidx]


@njit(parallel=True, fastmath=True)
def normalized_cumsum(data: NDArray) -> NDArray:
    """
    Calculate the normalized cumulative sum for an array.

    :param data: np.array
        Data to normalize
    :return: np.array
        Normalized cumulative sum
    """
    result = np.zeros(data.shape)
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            for p in range(data.shape[2]):
                result[i, j, p] = normalized_cumsum_pixel(data[i, j, p])
    return result


@njit
def normalized_cumsum_pixel(pixel: NDArray) -> NDArray:
    """
    Calculate the normalized cumulative sum of a single pixel.

    :param pixel: np.array
        Pixel data
    :return: np.array
        Normalized cumulative sum of the pixel
    """
    pixel = np.cumsum(pixel - 1)
    pixel -= np.min(pixel)
    max_val = np.max(pixel)
    if max_val > 0:
        pixel /= max_val
    return pixel
