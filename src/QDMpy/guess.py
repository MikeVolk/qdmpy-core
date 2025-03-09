"""
Module for guessing models and initial fit parameters for ODMR data.

This module provides functionality to automatically determine the appropriate model
for ODMR data and estimate initial fit parameters based on statistical analysis of
the data. It includes methods to guess the number of peaks, peak centers, widths,
contrasts, and other model parameters.

The module operates primarily on 4D numpy arrays containing ODMR data with dimensions:
(n_polarity, n_freq_range, n_frequencies, n_pixels).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.signal import find_peaks

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

from QDMpy.constants import DEFAULT_VMIN, DEFAULT_VMAX, PROMINENCE
from QDMpy.models import ModelRegistry
from QDMpy.exceptions import ModelGuessNotPossible

if TYPE_CHECKING:
    from QDMpy.models import Model

LOG = logging.getLogger(__name__)

@njit(fastmath=True)  # Remove parallel=True which may cause issues in test environment
def normalize_pixel(pixel: NDArray) -> NDArray:
    """
    Normalize a pixel's cumulative sum.

    Args:
        pixel: 1D array of intensity values for a single pixel.

    Returns:
        NDArray: The normalized cumulative sum of the pixel data.
    """
    pixel = np.cumsum(pixel - 1)  # Cumulative sum
    pixel -= np.min(pixel)  # Ensure non-negativity
    max_val = np.max(pixel)
    return pixel / max_val if max_val > 0 else pixel


def validate_array(data: NDArray, expected_dim: int, name: str) -> None:
    """
    Validate that an array has the expected number of dimensions.

    Args:
        data: The array to validate.
        expected_dim: The expected number of dimensions.
        name: The name of the array for error messages.

    Raises:
        ValueError: If the array does not have the expected number of dimensions.
    """
    if data.ndim != expected_dim:
        raise ValueError(
            f"{name} must have {expected_dim} dimensions. Got {data.ndim}."
        )


def guess_model(data: NDArray) -> Model:
    """
    Automatically determine the best fitting model for ODMR data.

    This function analyzes the number of peaks in the ODMR data and selects
    an appropriate model (ESR14N, ESR15N, or ESRSINGLE) based on the results.

    Args:
        data: 4D ODMR data array (n_polarity, n_range, n_frequencies, n_pixels).

    Returns:
        Model: An instance of the appropriate model.

    Raises:
        ModelGuessNotPossible: If the model cannot be reliably determined.
    """
    LOG.info("Trying to detect best fitting model for ODMR data.")
    n_peaks, doubt, _ = guess_n_peaks(data)

    if not doubt:
        model = get_model_by_peaks(n_peaks)
        LOG.info(f"Detected model: {model.name}")
        return model
    else:
        raise ModelGuessNotPossible(
            "Guessing the model is not possible. Please select model manually."
        )


def guess_n_peaks(data: NDArray) -> Tuple[int, bool, List[NDArray]]:
    """
    Estimate the number of peaks in ODMR data.

    This function analyzes the ODMR data to determine the number of resonance peaks
    by finding the negative peaks in the median value across pixels. It also assesses
    confidence in the peak count by checking the standard deviation of peak counts.

    Args:
        data: 4D ODMR data array (n_polarity, n_range, n_frequencies, n_pixels).

    Returns:
        Tuple containing:
            - int: Estimated number of peaks.
            - bool: Whether there is doubt in the estimate (True if uncertain).
            - List[NDArray]: Indices of detected peaks for each (polarity, frequency range) combination.

    Raises:
        ValueError: If the data array does not have 4 dimensions.
    """
    validate_array(data, 4, "data")
    median_data = np.median(data, axis=3)
    indices = [
        find_peaks(-median_data[p, f], prominence=PROMINENCE)[0]
        for p, f in np.ndindex(*data.shape[:2])
    ]
    n_peaks = int(np.round(np.mean([len(idx) for idx in indices])))
    doubt = np.std([len(idx) for idx in indices]) != 0
    return n_peaks, doubt, indices


def get_model_by_peaks(n_peaks: int) -> Model:
    """
    Retrieve the model instance dynamically based on the number of peaks.

    Args:
        n_peaks: Number of peaks detected in the data.

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


def guess_initial_fit_parameters(data: NDArray, freq: NDArray, model: Model) -> NDArray:
    """
    Guess initial fit parameters based on the selected model.

    Args:
        data: 4D array of the data to fit (e.g., ODMR data).
        freq: 1D array of the frequencies corresponding to the data.
        model: An instance of the selected model.

    Returns:
        NDArray: Initial fit parameters as a 3D array of shape
                 (n_pol, n_freq_range, n_params).

    Raises:
        ValueError: If a parameter type has no defined guess method.
    """
    # Define parameter guessers for each parameter type
    parameter_guessers = {
        "center": lambda: guess_center(data, freq),
        "contrast": lambda: guess_contrast(data),
        "width": lambda: guess_width(data, freq, DEFAULT_VMIN, DEFAULT_VMAX),
        "offset": lambda: np.ones((
            data.shape[0],
            data.shape[1],
            data.shape[3],
        )),  # Default offset is 1
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


@njit(fastmath=True)  # Simplify decorator for test compatibility
def guess_contrast(data: NDArray) -> NDArray:
    """
    Estimate the contrast for each pixel in the ODMR data.

    This function calculates the contrast (amplitude) of the ODMR signal
    for each pixel by finding the difference between maximum and minimum values.

    Args:
        data: 4D ODMR data array (n_polarity, n_range, n_frequencies, n_pixels).

    Returns:
        NDArray: 3D array of contrast values (n_polarity, n_range, n_pixels).
    """
    amp = np.zeros((data.shape[0], data.shape[1], data.shape[3]))
    for polarity in range(data.shape[0]):
        for freq_range in range(data.shape[1]):
            for pixel in range(data.shape[3]):  # Changed prange to range and fixed index
                amp[polarity, freq_range, pixel] = guess_contrast_pixel(
                    data[polarity, freq_range, :, pixel]
                )
    return amp


@njit(fastmath=True)
def guess_contrast_pixel(pixel: NDArray) -> float:
    """
    Estimate the contrast for a single pixel.

    This function calculates the contrast as the absolute fractional difference
    between the maximum and minimum values in the pixel data.

    Args:
        pixel: 1D array of intensity values for a single pixel.

    Returns:
        float: Estimated contrast value.
    """
    mx, mn = np.nanmax(pixel), np.nanmin(pixel)
    return 0 if mx == 0 else abs((mx - mn) / mx)


@njit(fastmath=True)  # Simplify decorator for test compatibility
def guess_center(data: NDArray, freq: NDArray) -> NDArray:
    """
    Guess the center frequency of ODMR data.

    Args:
        data: 4D ODMR data (n_polarity, n_range, n_frequencies, n_pixels).
        freq: 1D frequency range corresponding to the data.

    Returns:
        NDArray: 3D array of center frequencies (n_polarity, n_range, n_pixels).
    """
    centers = np.zeros((
        data.shape[0],
        data.shape[1],
        data.shape[3],
    ))  # Result shape: (n_polarity, n_range, n_pixels)
    for p in range(data.shape[0]):
        for r in range(data.shape[1]):
            for px in range(data.shape[3]):  # Changed prange to range
                centers[p, r, px] = guess_center_pixel(data[p, r, :, px], freq)
    return centers


@njit(fastmath=True)
def guess_center_pixel(pixel: NDArray, freq: NDArray) -> float:
    """
    Guess the center frequency of a single pixel.

    Args:
        pixel: 1D array of intensity values for a single pixel.
        freq: 1D array of frequency values.

    Returns:
        float: Guessed center frequency.
    """
    normalized = normalize_pixel(pixel)  # Normalize the pixel data
    idx = np.argmin(np.abs(normalized - 0.5))  # Find the index closest to the center
    return freq[idx]


@njit(fastmath=True)  # Simplify decorator for test compatibility
def guess_width(data: NDArray, freq: NDArray, vmin: float, vmax: float) -> NDArray:
    """
    Guess the width of ODMR resonance peaks.

    Args:
        data: 4D ODMR data (n_polarity, n_range, n_frequencies, n_pixels).
        freq: 1D frequency range corresponding to the data.
        vmin: Minimum normalized cumsum value.
        vmax: Maximum normalized cumsum value.

    Returns:
        NDArray: 3D array of widths (n_polarity, n_range, n_pixels).
    """
    widths = np.zeros((
        data.shape[0],
        data.shape[1],
        data.shape[3],
    ))  # Result shape: (n_polarity, n_range, n_pixels)
    for p in range(data.shape[0]):
        for r in range(data.shape[1]):
            for px in range(data.shape[3]):  # Changed prange to range
                widths[p, r, px] = guess_width_pixel(
                    data[p, r, :, px], freq, vmin, vmax
                )
    return widths


@njit(fastmath=True)
def guess_width_pixel(pixel: NDArray, freq: NDArray, vmin: float, vmax: float) -> float:
    """
    Guess the width of a single pixel.

    Args:
        pixel: 1D array of intensity values for a single pixel.
        freq: 1D array of frequency values.
        vmin: Minimum normalized cumsum value.
        vmax: Maximum normalized cumsum value.

    Returns:
        float: Estimated width of the resonance peak.
    """
    normalized = normalize_pixel(pixel)  # Normalize the pixel data
    lidx = np.argmin(np.abs(normalized - vmin))  # Index closest to vmin
    ridx = np.argmin(np.abs(normalized - vmax))  # Index closest to vmax
    # Always return a positive width by taking the absolute difference
    return abs(freq[ridx] - freq[lidx])


if __name__ == "__main__":
    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import (
        ODMRProcessorManager,
        BinningProcessor,
    )
    from QDMpy.odmr.odmr import ODMR
    from QDMpy.models import ModelRegistry

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
    # print(f"Guessed initial fit parameters: {fit_parameters}")
    # print(fit_parameters[0, 0, 100])
    # # Step 5: Apply model-specific calculations
    # calculated_data = model.func(freqs, fit_parameters[0, 0, 100])
    # print(f"Calculated data using model {model.name}: {calculated_data}")

    # # Step 6: Access processed data and metadata
    # print("Processed Data Shape:", odmr.processed_data.shape)
    # print("Metadata:", odmr.processed_data.metadata)
