"""Module for guessing models and initial fit parameters for ODMR data.

The numba-jitted functions in this module operate on 4D numpy arrays with the
convention: (n_polarity, n_freq_range, n_pixel, n_frequency).

Higher-level functions that accept xr.DataArray (or ODMRData) extract numpy
arrays at the boundary before calling into numba.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numba import njit, prange
from numpy.typing import NDArray
from scipy.signal import find_peaks

from qdmpy.exceptions import (
    DataShapeError,
    DataValidationError,
    ModelNotFoundError,
)
from qdmpy.fitting.models import ModelRegistry

if TYPE_CHECKING:
    from qdmpy.fitting.models import Model

# Prominence threshold as a fraction of the spectral range (max - min).
# The outer hyperfine peaks of ESR14N sit at ~7-10% of the spectral range,
# so 3% gives comfortable headroom while rejecting noise.
_RELATIVE_PROMINENCE = 0.03

# Fraction of (pol, frange) combinations that must agree on peak count
# before doubt is cleared. 0.6 tolerates one outlier in a 2-pol x 2-frange
# dataset (3/4 = 75% >= 60%).
_DETECTION_CONFIDENCE_THRESHOLD = 0.6


@njit(fastmath=True)
def normalize_pixel(pixel: NDArray) -> NDArray:  # pragma: no cover
    """Normalize a pixel's cumulative sum.

    Estimates the off-resonance baseline from the mean of the first and last
    10% of frequency points, then subtracts it before computing the cumulative
    sum. This is correct for both max-normalized data (where the baseline is
    exactly 1.0) and mean-normalized data (where the off-resonance level is
    slightly above 1.0, causing the old hardcoded subtraction of 1.0 to
    introduce a drift that distorts center and width estimates).

    Args:
        pixel: 1D array of intensity values for a single pixel.

    Returns:
        The normalized cumulative sum of the pixel data.
    """
    n = len(pixel)
    n_edge = max(1, n // 10)
    baseline = (np.mean(pixel[:n_edge]) + np.mean(pixel[n - n_edge :])) / 2
    pixel = np.cumsum(pixel - baseline)
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


def _relative_prominence(spectrum: NDArray) -> float:
    """Compute a per-spectrum prominence threshold as a fraction of its range.

    Args:
        spectrum: 1D array of intensity values.

    Returns:
        Prominence threshold in the same units as the spectrum.
    """
    spectral_range = float(np.max(spectrum) - np.min(spectrum))
    return max(spectral_range * _RELATIVE_PROMINENCE, 1e-6)


def guess_model(data: NDArray) -> Model:
    """Automatically determine the best fitting model for ODMR data.

    Uses majority-vote detection: the most common peak count across all
    (polarity, freq_range) combinations is returned. A warning is logged when
    agreement is below the confidence threshold, but a model is always returned
    so that auto-fitting can proceed.

    Args:
        data: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        An instance of the appropriate model.
    """
    logger.info("Trying to detect best fitting model for ODMR data.")
    n_peaks, doubt, _ = guess_n_peaks(data)
    model = get_model_by_peaks(n_peaks)
    if doubt:
        logger.warning(
            "Low-confidence model detection: using {} ({} peaks) "
            "as best guess. Verify with plot_model_detection() and set model_name "
            "manually if incorrect.",
            model.name,
            n_peaks,
        )
    else:
        logger.info("Detected model: {}", model.name)
    return model


def guess_n_peaks(data: NDArray) -> tuple[int, bool, list[NDArray]]:
    """Estimate the number of peaks in ODMR data via majority vote.

    Takes the median spectrum across pixels for each (polarity, freq_range)
    combination, detects dips using a per-spectrum relative prominence
    threshold, then picks the most common count (mode). Doubt is set when
    fewer than _DETECTION_CONFIDENCE_THRESHOLD of combinations agree.

    Args:
        data: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        Tuple of (n_peaks, doubt, peak_indices_list).
    """
    validate_array(data, 4, "data")
    median_data = np.median(data, axis=2)  # (n_pol, n_frange, n_freq)
    indices = []
    for p, f in np.ndindex(*data.shape[:2]):
        spectrum = median_data[p, f]
        prominence = _relative_prominence(spectrum)
        peaks = find_peaks(-spectrum, prominence=prominence)[0]
        indices.append(peaks)

    counts = [len(idx) for idx in indices]
    mode_count, mode_freq = Counter(counts).most_common(1)[0]
    confidence = mode_freq / len(counts)
    doubt = confidence < _DETECTION_CONFIDENCE_THRESHOLD
    return mode_count, doubt, indices


def plot_model_detection(
    data: NDArray,
    freq: NDArray | None = None,
) -> None:
    """Plot the median spectra used for model detection with detected peaks marked.

    Useful for visually verifying the auto-detection result, especially when
    doubt is flagged.

    Args:
        data: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).
        freq: Optional 2D frequency array (n_frange, n_freq) in GHz. If None,
              frequency index is used on the x-axis.
    """
    from qdmpy.plotting import plot_model_detection as _plot

    _plot(data, freq)


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
def top3_contrast(data: NDArray) -> NDArray:  # pragma: no cover
    """Estimate contrast from the top-3 and bottom-3 intensity values per pixel.

    max = mean of 3 largest values, min = mean of 3 smallest values.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape
    total = n_pol * n_frange * n_pixel

    amp = np.zeros((n_pol, n_frange, n_pixel))

    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)

        # initialize top-3 and bottom-3 trackers
        max1 = max2 = max3 = -np.inf
        min1 = min2 = min3 = np.inf

        for k in range(n_freq):
            v = data[p, r, px, k]
            if np.isnan(v):
                continue

            # update max trackers
            if v > max1:
                max3 = max2
                max2 = max1
                max1 = v
            elif v > max2:
                max3 = max2
                max2 = v
            elif v > max3:
                max3 = v

            # update min trackers
            if v < min1:
                min3 = min2
                min2 = min1
                min1 = v
            elif v < min2:
                min3 = min2
                min2 = v
            elif v < min3:
                min3 = v

        # mean of top 3 / bottom 3
        mx = (max1 + max2 + max3) / 3.0
        mn = (min1 + min2 + min3) / 3.0

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
def argmin_center(data: NDArray, freq: NDArray) -> NDArray:  # pragma: no cover
    """Guess center frequency as the frequency of the deepest dip per pixel.

    Unlike cumsum_center, this works correctly even when the resonance is
    shifted to the edge of the frequency range (strong B111 fields).

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
        centers[p, r, px] = freq[r, np.argmin(data[p, r, px])]
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


@njit(parallel=True, fastmath=True)
def halfpower_width(data: NDArray, freq: NDArray) -> NDArray:  # pragma: no cover
    """Estimate envelope HWHM from half-power points of each pixel spectrum.

    For each pixel, finds the deepest dip, computes the half-depth level,
    then searches left and right from the minimum for the crossing points.
    Returns FWHM / 2 (= HWHM of the absorption envelope).

    When the dip is near one edge of the frequency range (strong B111
    shift), the edge-mean baseline from that side is contaminated by the
    dip itself. To handle this, the baseline is estimated from whichever
    edge is farther from the dip minimum. If both sides fail (dip spans
    the full range), falls back to the spectrum maximum.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: 2D frequency array (n_frange, n_freq).

    Returns:
        3D array (n_pol, n_frange, n_pixel) of HWHM values in GHz.
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape
    total = n_pol * n_frange * n_pixel
    hwhm = np.zeros((n_pol, n_frange, n_pixel))
    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)
        spectrum = data[p, r, px]

        # Find minimum (deepest dip)
        min_val = spectrum[0]
        min_idx = 0
        for i in range(1, n_freq):
            if spectrum[i] < min_val:
                min_val = spectrum[i]
                min_idx = i

        # Baseline: use the edge farther from the dip to avoid
        # contamination when the dip is near one edge.
        n_edge = max(1, n_freq // 10)
        left_bl = 0.0
        for i in range(n_edge):
            left_bl += spectrum[i]
        left_bl /= n_edge

        right_bl = 0.0
        for i in range(n_freq - n_edge, n_freq):
            right_bl += spectrum[i]
        right_bl /= n_edge

        # Pick the edge farther from the dip
        baseline = right_bl if min_idx < n_freq // 2 else left_bl

        # Half-depth level
        half_depth = (baseline + min_val) / 2.0

        # Search left from minimum for crossing
        left_idx = min_idx
        for i in range(min_idx - 1, -1, -1):
            if spectrum[i] >= half_depth:
                left_idx = i
                break

        # Search right from minimum for crossing
        right_idx = min_idx
        for i in range(min_idx + 1, n_freq):
            if spectrum[i] >= half_depth:
                right_idx = i
                break

        # FWHM / 2 = HWHM
        fwhm = abs(freq[r, right_idx] - freq[r, left_idx])
        hwhm[p, r, px] = fwhm / 2.0

    return hwhm
