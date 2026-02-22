"""Benchmark competing implementations of ODMR initial-parameter guessing.

Implementations compared
------------------------
1. numba_old      — old production code (3 separate @njit loops, prange only over n_pixel,
                    normalize_pixel called twice per pixel — once in center, once in width)
2. numba_new      — NEW production code (same 3 separate functions, but flat prange over
                    n_pol*n_frange*n_pixel; normalize_pixel still called twice per pixel)
3. numba_combined — single kernel: prange over n_pixel, all 3 params, normalize_pixel once
4. numba_flat     — single kernel: flat prange over n_pol*n_frange*n_pixel, normalize once
5. numpy          — fully vectorized NumPy, no Numba, normalize computed once for whole array
6. fft            — FFT cross-correlation centre estimate (different algorithm entirely)

Note: np.unravel_index and np.nditer are not supported inside Numba @njit, so the flat-index
pattern (px = idx % n_pixel; r = (idx // n_pixel) % n_frange; p = idx // (n_pixel * n_frange))
is the standard Numba idiom for iterating a multi-dimensional index space in prange.

Usage
-----
    uv run python scripts/bench_guess.py [data_folder] [--bin FACTOR] [--runs N] [--no-verify]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Implementation 1: old Numba (mirrors production guess.py BEFORE QEP-024)
#   — nested loops, prange only at innermost (n_pixel level)
#   — normalize_pixel called twice per pixel (once for center, once for width)
# ---------------------------------------------------------------------------


@njit(fastmath=True)
def _normalize_pixel_nb(pixel: NDArray) -> NDArray:  # pragma: no cover
    cs = np.cumsum(pixel - 1.0)
    cs -= np.min(cs)
    mx = np.max(cs)
    return cs / mx if mx > 0 else cs


@njit(parallel=True, fastmath=True)
def _contrast_nb(data: NDArray) -> NDArray:  # pragma: no cover
    n_pol, n_frange, n_pixel, _ = data.shape
    out = np.zeros((n_pol, n_frange, n_pixel))
    for p in range(n_pol):
        for r in range(n_frange):
            for px in prange(n_pixel):  # type: ignore[not-iterable]
                mx = np.nanmax(data[p, r, px])
                mn = np.nanmin(data[p, r, px])
                out[p, r, px] = 0.0 if mx == 0.0 else abs((mx - mn) / mx)
    return out


@njit(parallel=True, fastmath=True)
def _center_nb(data: NDArray, freq: NDArray) -> NDArray:  # pragma: no cover
    n_pol, n_frange, n_pixel, _ = data.shape
    out = np.zeros((n_pol, n_frange, n_pixel))
    for p in range(n_pol):
        for r in range(n_frange):
            for px in prange(n_pixel):  # type: ignore[not-iterable]
                norm = _normalize_pixel_nb(data[p, r, px])
                out[p, r, px] = freq[r, np.argmin(np.abs(norm - 0.5))]
    return out


@njit(parallel=True, fastmath=True)
def _width_nb(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> NDArray:  # pragma: no cover
    n_pol, n_frange, n_pixel, _ = data.shape
    out = np.zeros((n_pol, n_frange, n_pixel))
    for p in range(n_pol):
        for r in range(n_frange):
            for px in prange(n_pixel):  # type: ignore[not-iterable]
                norm = _normalize_pixel_nb(data[p, r, px])  # redundant cumsum
                lidx = np.argmin(np.abs(norm - vmin))
                ridx = np.argmin(np.abs(norm - vmax))
                out[p, r, px] = abs(freq[r, ridx] - freq[r, lidx])
    return out


def guess_numba_old(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> tuple[NDArray, NDArray, NDArray]:
    contrast = _contrast_nb(data)
    center = _center_nb(data, freq)
    width = _width_nb(data, freq, vmin, vmax)
    return contrast, center, width


# ---------------------------------------------------------------------------
# Implementation 2: new production Numba (QEP-024) — calls the real
#   production functions from qdmpy_core.fitting.guess. Same 3 separate functions
#   but each uses flat prange over n_pol*n_frange*n_pixel.
#   normalize_pixel still called twice per pixel (in center + width).
# ---------------------------------------------------------------------------


def guess_numba_new(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> tuple[NDArray, NDArray, NDArray]:
    from qdmpy_core.fitting.guess import cumsum_center, cumsum_contrast, cumsum_width

    contrast = cumsum_contrast(data)
    center = cumsum_center(data, freq)
    width = cumsum_width(data, freq, vmin, vmax)
    return contrast, center, width


# ---------------------------------------------------------------------------
# Implementation 3: Numba combined — single prange over pixel, normalize once
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True)
def _guess_all_nb_combined(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> tuple[NDArray, NDArray, NDArray]:  # pragma: no cover
    n_pol, n_frange, n_pixel, _ = data.shape
    contrast = np.zeros((n_pol, n_frange, n_pixel))
    center = np.zeros((n_pol, n_frange, n_pixel))
    width = np.zeros((n_pol, n_frange, n_pixel))
    for p in range(n_pol):
        for r in range(n_frange):
            for px in prange(n_pixel):  # type: ignore[not-iterable]
                pixel = data[p, r, px]
                mx = np.nanmax(pixel)
                mn = np.nanmin(pixel)
                contrast[p, r, px] = 0.0 if mx == 0.0 else abs((mx - mn) / mx)
                norm = _normalize_pixel_nb(pixel)
                center[p, r, px] = freq[r, np.argmin(np.abs(norm - 0.5))]
                lidx = np.argmin(np.abs(norm - vmin))
                ridx = np.argmin(np.abs(norm - vmax))
                width[p, r, px] = abs(freq[r, ridx] - freq[r, lidx])
    return contrast, center, width


def guess_numba_combined(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> tuple[NDArray, NDArray, NDArray]:
    return _guess_all_nb_combined(data, freq, vmin, vmax)


# ---------------------------------------------------------------------------
# Implementation 2b: Numba flat — single prange over all (pol, frange, pixel)
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True)
def _guess_all_nb_flat(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> tuple[NDArray, NDArray, NDArray]:  # pragma: no cover
    n_pol, n_frange, n_pixel, _ = data.shape
    total = n_pol * n_frange * n_pixel
    contrast = np.zeros((n_pol, n_frange, n_pixel))
    center = np.zeros((n_pol, n_frange, n_pixel))
    width = np.zeros((n_pol, n_frange, n_pixel))
    for idx in prange(total):  # type: ignore[not-iterable]
        px = idx % n_pixel
        r = (idx // n_pixel) % n_frange
        p = idx // (n_pixel * n_frange)
        pixel = data[p, r, px]
        mx = np.nanmax(pixel)
        mn = np.nanmin(pixel)
        contrast[p, r, px] = 0.0 if mx == 0.0 else abs((mx - mn) / mx)
        norm = _normalize_pixel_nb(pixel)
        center[p, r, px] = freq[r, np.argmin(np.abs(norm - 0.5))]
        lidx = np.argmin(np.abs(norm - vmin))
        ridx = np.argmin(np.abs(norm - vmax))
        width[p, r, px] = abs(freq[r, ridx] - freq[r, lidx])
    return contrast, center, width


def guess_numba_flat(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> tuple[NDArray, NDArray, NDArray]:
    return _guess_all_nb_flat(data, freq, vmin, vmax)


# ---------------------------------------------------------------------------
# Implementation 3: vectorized NumPy
# ---------------------------------------------------------------------------


def _normalize_all_np(data: NDArray) -> NDArray:
    """Compute normalized cumsum for the entire array in one shot.

    Args:
        data: (n_pol, n_frange, n_pixel, n_freq)

    Returns:
        (n_pol, n_frange, n_pixel, n_freq) normalized to [0, 1] per pixel
    """
    cs = np.cumsum(data - 1.0, axis=-1)
    mn = cs.min(axis=-1, keepdims=True)
    rng = cs.max(axis=-1, keepdims=True) - mn
    return (cs - mn) / np.where(rng > 0, rng, 1.0)


def guess_numpy(
    data: NDArray, freq: NDArray, vmin: float, vmax: float
) -> tuple[NDArray, NDArray, NDArray]:
    # contrast — no cumsum
    mx = np.nanmax(data, axis=-1)
    mn = np.nanmin(data, axis=-1)
    safe_mx = np.where(mx != 0, mx, 1.0)
    contrast = np.where(mx != 0, np.abs((mx - mn) / safe_mx), 0.0)

    # single cumsum pass shared by center and width
    normalized = _normalize_all_np(data)  # (pol, frange, px, freq)

    # advanced index: freq[frange_idx, pixel_argmin] → (pol, frange, px)
    frange_idx = np.arange(data.shape[1])[np.newaxis, :, np.newaxis]  # (1, fr, 1)

    center_idx = np.argmin(np.abs(normalized - 0.5), axis=-1)
    center = freq[frange_idx, center_idx]

    lidx = np.argmin(np.abs(normalized - vmin), axis=-1)
    ridx = np.argmin(np.abs(normalized - vmax), axis=-1)
    width = np.abs(freq[frange_idx, ridx] - freq[frange_idx, lidx])

    return contrast, center, width


# ---------------------------------------------------------------------------
# Implementation 4: FFT cross-correlation (model-aware matched filter)
#
# Center: cross-correlate inverted spectrum with a Lorentzian (or triplet)
#         template. Peak of cross-correlation = center shift. Fully
#         vectorized over all pixels with a single rfft call per freq-range.
#
# Width:  the FFT of a Lorentzian L(f,w) decays as exp(-2π·w·|ξ|). We
#         estimate w from the slope of log|FFT(inverted spectrum)| in the
#         high-frequency bins (linear regression on log-magnitude).
#
# Contrast: same (max-min)/max as current code — no FFT needed.
# ---------------------------------------------------------------------------


def _lorentzian_template(freq: NDArray, center: float, width: float) -> NDArray:
    return 1.0 / (1.0 + ((freq - center) / width) ** 2)


def _build_template(freq_range: NDArray, n_peaks: int, width: float = 0.004) -> NDArray:
    """Build a model-aware Lorentzian template centred in the freq range.

    Args:
        freq_range: 1-D frequency array for one frange.
        n_peaks: 1 (ESRSINGLE), 2 (ESR15N), or 3 (ESR14N).
        width: Lorentzian half-width in GHz (default 4 MHz).

    Returns:
        Normalised template array, same length as freq_range.
    """
    from qdmpy_core.constants import AHYP_14N, AHYP_15N

    f0 = (freq_range[0] + freq_range[-1]) / 2.0
    if n_peaks == 3:
        offsets = np.array([-AHYP_14N, 0.0, AHYP_14N])
    elif n_peaks == 2:
        offsets = np.array([-AHYP_15N / 2, AHYP_15N / 2])
    else:
        offsets = np.array([0.0])

    template = sum(_lorentzian_template(freq_range, f0 + dk, width) for dk in offsets)
    return template / template.max()


def guess_fft(
    data: NDArray, freq: NDArray, vmin: float, vmax: float, n_peaks: int = 3
) -> tuple[NDArray, NDArray, NDArray]:
    """Estimate initial parameters using FFT cross-correlation.

    Args:
        data: (n_pol, n_frange, n_pixel, n_freq)
        freq: (n_frange, n_freq) in GHz
        vmin: unused (kept for uniform interface)
        vmax: unused (kept for uniform interface)
        n_peaks: number of Lorentzian peaks in the model (1, 2, or 3)

    Returns:
        (contrast, center, width) each (n_pol, n_frange, n_pixel)
    """
    n_pol, n_frange, n_pixel, n_freq = data.shape

    # zero-pad to next power-of-2 >= 2*n_freq for circular-correlation safety
    n_fft = int(2 ** np.ceil(np.log2(2 * n_freq)))

    contrast = np.empty((n_pol, n_frange, n_pixel))
    center = np.empty((n_pol, n_frange, n_pixel))
    width = np.empty((n_pol, n_frange, n_pixel))

    # high-frequency bins used for width estimation (top 40% of rfft output)
    n_rfft = n_fft // 2 + 1
    xi = np.arange(n_rfft) / n_fft  # normalised frequencies
    hi_mask = xi > 0.3  # use upper 40% of spectrum
    xi_hi = xi[hi_mask]

    np.arange(n_frange)[np.newaxis, :, np.newaxis]  # (1, fr, 1)

    for r in range(n_frange):
        f = freq[r]  # (n_freq,)
        df = float(f[1] - f[0])
        f0 = float((f[0] + f[-1]) / 2.0)

        # --- template (built once per freq-range) ---
        tmpl = _build_template(f, n_peaks)
        T = np.fft.rfft(tmpl, n=n_fft)  # (n_rfft,)

        # --- inverted spectra: dips → peaks, shape (n_pol, n_pixel, n_freq) ---
        inv = 1.0 - data[:, r, :, :]

        # --- contrast from raw (max-min)/max ---
        mx = np.nanmax(data[:, r, :, :], axis=-1)
        mn = np.nanmin(data[:, r, :, :], axis=-1)
        contrast[:, r, :] = np.where(mx != 0, np.abs((mx - mn) / np.where(mx != 0, mx, 1.0)), 0.0)

        # --- FFT of all pixels at once (n_pol, n_pixel, n_rfft) ---
        S = np.fft.rfft(inv, n=n_fft, axis=-1)

        # --- center via cross-correlation ---
        corr = np.fft.irfft(S * np.conj(T)[np.newaxis, np.newaxis, :], n=n_fft, axis=-1)
        # take only first n_freq lags (avoid wrap-around artefacts)
        shift_idx = np.argmax(corr[:, :, :n_freq], axis=-1)  # (n_pol, n_pixel)
        # shifts > n_freq//2 are negative (template shifted left)
        shift_idx = np.where(shift_idx > n_freq // 2, shift_idx - n_freq, shift_idx)
        center[:, r, :] = f0 + shift_idx * df

        # --- width from log-magnitude decay of FFT ---
        # |FT{Lorentzian(w)}| ∝ exp(-2π·w·|ξ|) → log|FT| = const - 2π·w·ξ
        # fit slope via mean: w ≈ -d(log|S|)/d(ξ) / (2π)
        log_mag = np.log(np.abs(S[:, :, hi_mask]) + 1e-12)  # (n_pol, n_pixel, n_hi)
        # least-squares slope: Σ(xi·log_mag) / Σ(xi²)
        slope = np.sum(xi_hi * log_mag, axis=-1) / np.sum(xi_hi**2)
        w_est = np.clip(-slope / (2 * np.pi), 0.001, 0.02)  # clip to 1–20 MHz
        width[:, r, :] = w_est

    return contrast, center, width


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def warmup(impl_fn, data, freq, vmin, vmax, n: int = 3) -> None:
    for _ in range(n):
        impl_fn(data, freq, vmin, vmax)


def bench(impl_fn, data, freq, vmin, vmax, n_runs: int) -> list[float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        impl_fn(data, freq, vmin, vmax)
        times.append(time.perf_counter() - t0)
    return times


def verify_agreement(
    ref: tuple[NDArray, NDArray, NDArray],
    other: tuple[NDArray, NDArray, NDArray],
    name: str,
    tol: float = 1e-5,
) -> None:
    labels = ("contrast", "center", "width")
    for _label, r, o in zip(labels, ref, other, strict=False):
        np.max(np.abs(r - o))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark guess implementations")
    parser.add_argument(
        "data_folder",
        nargs="?",
        default=str(Path.home() / "Documents" / "FOV18x"),
    )
    parser.add_argument("--bin", type=int, default=2, metavar="FACTOR")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical agreement check")
    args = parser.parse_args()

    # --- load data ---
    from loguru import logger

    logger.disable("QDMpy")

    from qdmpy_core.constants import DEFAULT_VMAX, DEFAULT_VMIN
    from qdmpy_core.odmr.data import ODMRData
    from qdmpy_core.odmr.io import MatlabLoader
    from qdmpy_core.odmr.manager import ODMR
    from qdmpy_core.odmr.processors import BinningProcessor, NormalizationProcessor

    loader = MatlabLoader(data_folder=args.data_folder)
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    if args.bin > 1:
        odmr.processor_manager.add_processor(BinningProcessor(bin_factor=args.bin))
    odmr.processor_manager.add_processor(NormalizationProcessor())
    odmr.process_data()

    processed = odmr.processed_data
    raw = processed.data.values  # (pol, frange, y, x, freq)
    n_pol, n_frange, h, w, n_freq = raw.shape
    data = raw.reshape(n_pol, n_frange, h * w, n_freq)
    freq = processed.frequencies  # (n_frange, n_freq)

    vmin, vmax = DEFAULT_VMIN, DEFAULT_VMAX

    # detect n_peaks from the data for FFT template
    from qdmpy_core.fitting.guess import guess_model

    detected_model = guess_model(data)
    n_peaks = detected_model.n_peaks

    import functools

    guess_fft_bound = functools.partial(guess_fft, n_peaks=n_peaks)

    # group: cumsum-based (apples-to-apples speed) vs fft (different algorithm)
    cumsum_impls: list[tuple[str, object]] = [
        ("numba_old", guess_numba_old),  # old production (nested prange)
        ("numba_new", guess_numba_new),  # NEW production (flat prange, separate fns)
        ("numba_combined", guess_numba_combined),  # single kernel, prange(n_pixel)
        ("numba_flat", guess_numba_flat),  # single kernel, flat prange
        ("numpy", guess_numpy),
    ]

    fft_impls: list[tuple[str, object]] = [
        ("fft", guess_fft_bound),
    ]
    implementations = cumsum_impls + fft_impls

    # --- warm up ---
    for name, fn in implementations:
        warmup(fn, data, freq, vmin, vmax)

    # --- numerical correctness: numba_new must match numba_old exactly ---
    ref_c, ref_ct, ref_w = guess_numba_old(data, freq, vmin, vmax)
    new_c, new_ct, new_w = guess_numba_new(data, freq, vmin, vmax)
    verify_agreement((ref_c, ref_ct, ref_w), (new_c, new_ct, new_w), "numba_new")

    # --- compare fft estimates vs cumsum (accuracy, not just numerical equality) ---
    _fft_c, _fft_ct, _fft_w = guess_fft_bound(data, freq, vmin, vmax)
    float(freq[0, 1] - freq[0, 0])

    # --- benchmark ---
    results: dict[str, list[float]] = {}
    for name, fn in implementations:
        results[name] = bench(fn, data, freq, vmin, vmax, args.runs)

    # --- report ---

    baseline_mean = float(np.mean(results["numba_old"]))
    for name, times in results.items():
        min(times) * 1e3
        float(np.mean(times)) * 1e3
        max(times) * 1e3
        speedup = baseline_mean / float(np.mean(times))
        marker = (
            "  ← baseline"
            if name == "numba_old"
            else (f"  {speedup:.1f}×" if speedup > 1 else f"  {speedup:.2f}×")
        )
        if name == "fft":
            marker += "  [different algorithm]"


if __name__ == "__main__":
    main()
