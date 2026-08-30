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


# NOTE: no fastmath here -- fastmath=True implies LLVM's no-NaN assumption,
# under which numba folds np.isnan(...) to False and the NaN guards below
# silently stop working. Verified: top3_contrast returned 0.0 contrast for
# any pixel holding a single NaN while it carried fastmath=True.
@njit
def normalize_pixel(pixel: NDArray) -> NDArray:  # pragma: no cover
    """Normalize a pixel's cumulative sum.

    Estimates the off-resonance baseline from the mean of the first and last
    10% of frequency points, then subtracts it before computing the cumulative
    sum. This is correct for both max-normalized data (where the baseline is
    exactly 1.0) and mean-normalized data (where the off-resonance level is
    slightly above 1.0, causing the old hardcoded subtraction of 1.0 to
    introduce a drift that distorts center and width estimates).

    NaN-safe: NaN samples are excluded from the baseline and treated as zero
    contribution in the cumulative sum. A plain ``np.mean`` over an edge
    containing one NaN returns NaN, which made the whole normalized curve NaN
    and silently collapsed downstream ``argmin`` lookups onto index 0.

    Args:
        pixel: 1D array of intensity values for a single pixel.

    Returns:
        The normalized cumulative sum of the pixel data. All-NaN input
        returns an array of zeros.
    """
    n = len(pixel)
    n_edge = max(1, n // 10)

    baseline_sum = 0.0
    baseline_count = 0
    for i in range(n_edge):
        if not np.isnan(pixel[i]):
            baseline_sum += pixel[i]
            baseline_count += 1
    for i in range(n - n_edge, n):
        if not np.isnan(pixel[i]):
            baseline_sum += pixel[i]
            baseline_count += 1

    if baseline_count == 0:
        return np.zeros(n, dtype=pixel.dtype)
    baseline = baseline_sum / baseline_count

    centred = np.empty(n, dtype=pixel.dtype)
    for i in range(n):
        centred[i] = 0.0 if np.isnan(pixel[i]) else pixel[i] - baseline

    result = np.cumsum(centred)
    result -= np.min(result)
    max_val = np.max(result)
    return result / max_val if max_val > 0 else result


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
    finite = spectrum[np.isfinite(spectrum)]
    if finite.size == 0:
        return 1e-6
    spectral_range = float(finite.max() - finite.min())
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
    # nanmedian, not median: a single NaN pixel would otherwise poison the
    # whole median spectrum for that (pol, frange) and hence model detection.
    median_data = np.nanmedian(data, axis=2)  # (n_pol, n_frange, n_freq)
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


# NOTE: no fastmath here -- fastmath=True implies LLVM's no-NaN assumption,
# under which numba folds np.isnan(...) to False and the NaN guards below
# silently stop working. Verified: top3_contrast returned 0.0 contrast for
# any pixel holding a single NaN while it carried fastmath=True.
@njit(parallel=True)
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


# NOTE: no fastmath here -- fastmath=True implies LLVM's no-NaN assumption,
# under which numba folds np.isnan(...) to False and the NaN guards below
# silently stop working. Verified: top3_contrast returned 0.0 contrast for
# any pixel holding a single NaN while it carried fastmath=True.
@njit(parallel=True)
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


# NOTE: no fastmath here -- fastmath=True implies LLVM's no-NaN assumption,
# under which numba folds np.isnan(...) to False and the NaN guards below
# silently stop working. Verified: top3_contrast returned 0.0 contrast for
# any pixel holding a single NaN while it carried fastmath=True.
@njit(parallel=True)
def argmin_center(data: NDArray, freq: NDArray) -> NDArray:  # pragma: no cover
    """Guess center frequency as the frequency of the deepest dip per pixel.

    Unlike cumsum_center, this works correctly even when the resonance is
    shifted to the edge of the frequency range (strong B111 fields).

    NaN-safe: ``np.argmin`` treats NaN as the minimum, so a single dead
    sample used to pin the centre guess to that arbitrary frequency. NaN
    samples are skipped; an all-NaN spectrum falls back to the midpoint of
    the frequency range.

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: 2D frequency array (n_frange, n_freq).

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape
    total = n_pol * n_frange * n_pixel
    centers = np.zeros((n_pol, n_frange, n_pixel))
    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)

        spectrum = data[p, r, px]
        min_val = np.inf
        min_idx = -1
        for i in range(n_freq):
            v = spectrum[i]
            if not np.isnan(v) and v < min_val:
                min_val = v
                min_idx = i

        if min_idx < 0:
            centers[p, r, px] = (freq[r, 0] + freq[r, n_freq - 1]) / 2.0
        else:
            centers[p, r, px] = freq[r, min_idx]
    return centers


@njit(parallel=True, fastmath=True)
def absorption_centroid(data: NDArray, freq: NDArray) -> NDArray:  # pragma: no cover
    """Guess center frequency using the absorption-weighted centroid per pixel.

    For each pixel the centroid is:

        absorption[i] = max(baseline - spectrum[i], 0)
        center = sum(freq[i] * absorption[i]) / sum(absorption[i])

    This is correct for all three supported models:
    - ESRSINGLE: single dip -> centroid == dip frequency (same as argmin).
    - ESR15N: two symmetric dips -> centroid lands at the midpoint = true center.
    - ESR14N: three dips -> centroid lands at the central dip = true center,
      even when outer-dip contrasts are unequal (centroid stays inside the
      absorption envelope, never +-AHYP away like argmin can be).

    Fallback: when sum(absorption) == 0 (flat or all-below-baseline spectrum),
    returns the midpoint of the frequency range for that (pol, frange).

    Args:
        data: 4D array (n_pol, n_frange, n_pixel, n_freq).
        freq: 2D frequency array (n_frange, n_freq).

    Returns:
        3D array (n_pol, n_frange, n_pixel).
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape
    total = n_pol * n_frange * n_pixel
    centers = np.zeros((n_pol, n_frange, n_pixel))
    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)

        spectrum = data[p, r, px]
        n_edge = max(1, n_freq // 10)

        # Estimate off-resonance baseline from first and last 10% of points
        left_sum = 0.0
        for i in range(n_edge):
            left_sum += spectrum[i]
        right_sum = 0.0
        for i in range(n_freq - n_edge, n_freq):
            right_sum += spectrum[i]
        baseline = (left_sum + right_sum) / (2.0 * n_edge)

        # Absorption-weighted centroid
        weight_sum = 0.0
        freq_sum = 0.0
        for i in range(n_freq):
            a = baseline - spectrum[i]
            if a > 0.0:
                weight_sum += a
                freq_sum += freq[r, i] * a

        if weight_sum > 0.0:
            centers[p, r, px] = freq_sum / weight_sum
        else:
            centers[p, r, px] = (freq[r, 0] + freq[r, n_freq - 1]) / 2.0
    return centers


# NOTE: no fastmath here -- fastmath=True implies LLVM's no-NaN assumption,
# under which numba folds np.isnan(...) to False and the NaN guards below
# silently stop working. Verified: top3_contrast returned 0.0 contrast for
# any pixel holding a single NaN while it carried fastmath=True.
@njit(parallel=True)
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


@njit
def _edge_baseline(spectrum: NDArray, min_idx: int) -> float:  # pragma: no cover
    """Off-resonance baseline from whichever spectrum edge is farther from the dip.

    The near edge is contaminated by the dip itself when the resonance is
    shifted towards it (strong B111), so the far edge is preferred. Falls back
    to the other edge when the preferred one holds no finite samples.

    Args:
        spectrum: 1D intensity values for one pixel.
        min_idx: Index of the deepest (finite) sample.

    Returns:
        The baseline level, or NaN when neither edge has a finite sample.
    """
    n_freq = len(spectrum)
    n_edge = max(1, n_freq // 10)

    left_sum = 0.0
    left_n = 0
    for i in range(n_edge):
        if not np.isnan(spectrum[i]):
            left_sum += spectrum[i]
            left_n += 1

    right_sum = 0.0
    right_n = 0
    for i in range(n_freq - n_edge, n_freq):
        if not np.isnan(spectrum[i]):
            right_sum += spectrum[i]
            right_n += 1

    prefer_right = min_idx < n_freq // 2
    if prefer_right and right_n > 0:
        return right_sum / right_n
    if not prefer_right and left_n > 0:
        return left_sum / left_n
    if right_n > 0:
        return right_sum / right_n
    if left_n > 0:
        return left_sum / left_n
    return np.nan


# NOTE: no fastmath here -- fastmath=True implies LLVM's no-NaN assumption,
# under which numba folds np.isnan(...) to False and the NaN guards below
# silently stop working. Verified: top3_contrast returned 0.0 contrast for
# any pixel holding a single NaN while it carried fastmath=True.
@njit(parallel=True)
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

        # Find minimum (deepest dip). NaN comparisons are always False, so
        # seeding from spectrum[0] pinned the minimum at index 0 whenever the
        # first sample was NaN; skip NaN explicitly instead.
        min_val = np.inf
        min_idx = -1
        for i in range(n_freq):
            v = spectrum[i]
            if not np.isnan(v) and v < min_val:
                min_val = v
                min_idx = i

        if min_idx < 0:
            hwhm[p, r, px] = 0.0
            continue

        baseline = _edge_baseline(spectrum, min_idx)
        if np.isnan(baseline):
            hwhm[p, r, px] = 0.0
            continue

        # Half-depth level
        half_depth = (baseline + min_val) / 2.0

        # Search outward from the minimum for the half-depth crossings. A NaN
        # sample is skipped rather than ending the search. When no crossing is
        # found the dip runs off that side of the window, so fall back to the
        # window edge -- collapsing to min_idx would report a zero width, which
        # is both wrong and outside every sane width constraint.
        left_idx = 0
        for i in range(min_idx - 1, -1, -1):
            if not np.isnan(spectrum[i]) and spectrum[i] >= half_depth:
                left_idx = i
                break

        right_idx = n_freq - 1
        for i in range(min_idx + 1, n_freq):
            if not np.isnan(spectrum[i]) and spectrum[i] >= half_depth:
                right_idx = i
                break

        # FWHM / 2 = HWHM
        fwhm = abs(freq[r, right_idx] - freq[r, left_idx])
        hwhm[p, r, px] = fwhm / 2.0

    return hwhm
