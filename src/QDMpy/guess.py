from __future__ import annotations

from numba import njit, prange
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING
import numpy as np
from scipy.signal import find_peaks
import logging

import os
import sys

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

from QDMpy.constants import DEFAULT_VMIN, DEFAULT_VMAX, PROMINENCE
from QDMpy.models import ModelRegistry

if TYPE_CHECKING:
    from QDMpy.models import Model

LOG = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def normalize_pixel(pixel: NDArray) -> NDArray:
    pixel = np.cumsum(pixel - 1)
    pixel -= np.min(pixel)
    max_val = np.max(pixel)
    return pixel / max_val if max_val > 0 else pixel


def validate_array(data: NDArray, expected_dim: int, name: str):
    if data.ndim != expected_dim:
        raise ValueError(
            f"{name} must have {expected_dim} dimensions. Got {data.ndim}."
        )


def guess_n_peaks(data: NDArray) -> tuple[int, bool, Any]:
    validate_array(data, 4, "data")
    median_data = np.median(data, axis=3)
    indices = [
        find_peaks(-median_data[p, f], prominence=PROMINENCE)[0]
        for p, f in np.ndindex(*data.shape[:2])
    ]
    n_peaks = int(np.round(np.mean([len(idx) for idx in indices])))
    doubt = np.std([len(idx) for idx in indices]) != 0
    return n_peaks, doubt, indices

def get_model_by_peaks(n_peaks: int):
    """
    Retrieve the model instance dynamically based on the number of peaks.

    Args:
        n_peaks (int): Number of peaks detected in the data.

    Returns:
        Model: The corresponding model instance.

    Raises:
        ValueError: If no model matches the given number of peaks.
    """
    for model_info in ModelRegistry.all().values():
        model_class = model_info["class"]
        model_instance = model_class()
        if model_instance.n_peaks == n_peaks:
            return model_instance
    raise ValueError(f"No model found for {n_peaks} peaks.")



def guess_initial_fit_parameters(
    data: NDArray, freq: NDArray, model: Model
) -> NDArray:
    """
    Guess initial fit parameters based on the selected model.

    Args:
        data (NDArray): 4D array of the data to fit (e.g., ODMR data).
        freq (NDArray): 1D array of the frequencies corresponding to the data.
        model (Model): An instance of the selected model.

    Returns:
        NDArray: Initial fit parameters as a 4D array of shape
                 (n_pol, n_freq_range, n_pixel, n_params).
    """
    # Define parameter guessers for each parameter type
    parameter_guessers = {
        "center": lambda: guess_center(data, freq),
        "contrast": lambda: guess_contrast(data),
        "width": lambda: guess_width(data, freq, DEFAULT_VMIN, DEFAULT_VMAX),
        "offset": lambda: np.ones((data.shape[0],data.shape[1],data.shape[3])),  # Default offset is 1
    }

    # Initialize list for parameter arrays
    fit_parameters = []

    # Guess each parameter defined in the model
    for param in model.parameters_unique:
        param_type = param.split("_")[0]  # Extract parameter type (e.g., 'width')
        LOG.info(f"Calculating initial guess for {param_type}")
        if param_type in parameter_guessers:
            fit_parameters.append(parameter_guessers[param_type]())
        else:
            raise ValueError(f"Parameter '{param}' has no defined guess method.")

    # Stack parameters along the last axis
    return np.stack(fit_parameters, axis=-1)



@njit(parallel=True, fastmath=True)
def guess_contrast(data: NDArray) -> NDArray:
    amp = np.zeros((data.shape[0],data.shape[1],data.shape[3]))
    for polarity in prange(data.shape[0]):
        for freq_range in range(data.shape[1]):
            for pixel in range(data.shape[2]):
                amp[polarity, freq_range, pixel] = guess_contrast_pixel(data[polarity, freq_range, :, pixel])
    return amp


@njit(fastmath=True)
def guess_contrast_pixel(pixel: NDArray) -> float:
    mx, mn = np.nanmax(pixel), np.nanmin(pixel)
    return 0 if mx == 0 else abs((mx - mn) / mx)

@njit(parallel=True, fastmath=True)
def guess_center(data: NDArray, freq: NDArray) -> NDArray:
    """
    Guess the center frequency of ODMR data.

    Args:
        data (NDArray): 4D ODMR data (n_polarity, n_range, n_frequencies, n_pixels).
        freq (NDArray): 1D frequency range corresponding to the data.

    Returns:
        NDArray: 3D array of center frequencies (n_polarity, n_range, n_pixels).
    """
    centers = np.zeros((data.shape[0],data.shape[1],data.shape[3]))  # Result shape: (n_polarity, n_range, n_pixels)
    for p in prange(data.shape[0]):
        for r in range(data.shape[1]):
            for px in range(data.shape[3]):
                centers[p, r, px] = guess_center_pixel(data[p, r, :, px], freq)
    return centers


@njit(fastmath=True)
def guess_center_pixel(pixel: NDArray, freq: NDArray) -> float:
    """
    Guess the center frequency of a single pixel.

    Args:
        pixel (NDArray): 1D array of intensity values for a single pixel.
        freq (NDArray): 1D array of frequency values.

    Returns:
        float: Guessed center frequency.
    """
    normalized = normalize_pixel(pixel)  # Normalize the pixel data
    idx = np.argmin(np.abs(normalized - 0.5))  # Find the index closest to the center
    return freq[idx]

@njit(parallel=True, fastmath=True)
def guess_width(data: NDArray, freq: NDArray, vmin: float, vmax: float) -> NDArray:
    """
    Guess the width of ODMR resonance peaks.

    Args:
        data (NDArray): 4D ODMR data (n_polarity, n_range, n_frequencies, n_pixels).
        freq (NDArray): 1D frequency range corresponding to the data.
        vmin (float): Minimum normalized cumsum value.
        vmax (float): Maximum normalized cumsum value.

    Returns:
        NDArray: 3D array of widths (n_polarity, n_range, n_pixels).
    """
    widths = np.zeros((data.shape[0],data.shape[1],data.shape[3]))  # Result shape: (n_polarity, n_range, n_pixels)
    for p in prange(data.shape[0]):
        for r in range(data.shape[1]):
            for px in range(data.shape[3]):
                widths[p, r, px] = guess_width_pixel(data[p, r, :, px], freq, vmin, vmax)
    return widths


@njit(fastmath=True)
def guess_width_pixel(pixel: NDArray, freq: NDArray, vmin: float, vmax: float) -> float:
    """
    Guess the width of a single pixel.

    Args:
        pixel (NDArray): 1D array of intensity values for a single pixel.
        freq (NDArray): 1D array of frequency values.
        vmin (float): Minimum normalized cumsum value.
        vmax (float): Maximum normalized cumsum value.

    Returns:
        float: Estimated width of the resonance peak.
    """
    normalized = normalize_pixel(pixel)  # Normalize the pixel data
    lidx = np.argmin(np.abs(normalized - vmin))  # Index closest to vmin
    ridx = np.argmin(np.abs(normalized - vmax))  # Index closest to vmax
    return freq[ridx] - freq[lidx]



@njit(fastmath=True)
def normalize_pixel(pixel: NDArray) -> NDArray:
    """
    Normalize a pixel's cumulative sum.

    Args:
        pixel (NDArray): 1D array of intensity values for a single pixel.

    Returns:
        NDArray: The normalized cumulative sum of the pixel data.
    """
    pixel = np.cumsum(pixel - 1)  # Cumulative sum
    pixel -= np.min(pixel)  # Ensure non-negativity
    max_val = np.max(pixel)
    return pixel / max_val if max_val > 0 else pixel



if __name__ == "__main__":
    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import (
        ODMRProcessorManager,
        NormalizationProcessor,
        BinningProcessor,
    )
    from QDMpy.odmr.odmr import ODMR
    from QDMpy.models import ModelRegistry
    from QDMpy.guess import guess_n_peaks, guess_initial_fit_parameters

    # Step 1: Load ODMR data
    loader = MatlabLoader(data_folder="/home/mike/git/QDMpy/tests/data")
    raw_data, scan_dims, freqs = loader.load()
    odmr_data = ODMRData(data=raw_data, scan_dimensions=scan_dims, frequencies=freqs)

    # Step 2: Initialize ODMR manager
    odmr = ODMR(odmr_data)

    # Step 3: Process the data (optional normalization and binning)
    processor_manager = ODMRProcessorManager()
    # processor_manager.add_processor(NormalizationProcessor(method="max"))
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.process_data()

    # Step 4: Guess the model and parameters
    n_peaks, doubt, _ = guess_n_peaks(odmr.processed_data.data)
    print(f"Guessed number of peaks: {n_peaks}, Doubt: {doubt}")

    model = get_model_by_peaks(n_peaks)
    fit_parameters = guess_initial_fit_parameters(
        odmr.processed_data.data, freqs, model
    )
    print(f"Guessed initial fit parameters: {fit_parameters}")
    print(fit_parameters.shape)
    # Step 5: Apply model-specific calculations
    calculated_data = model.func(freqs, fit_parameters[0,0,100])
    print(f"Calculated data using model {model.name}: {calculated_data}")

    # Step 6: Access processed data and metadata
    print("Processed Data Shape:", odmr.processed_data.shape)
    print("Metadata:", odmr.processed_data.metadata)
