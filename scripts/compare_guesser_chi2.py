"""Compare initial-parameter chi² between old (baseline=1.0) and new (edge-based) guessers.

The old ``normalize_pixel`` hardcoded ``pixel - 1``, which assumed the off-resonance
baseline is exactly 1.0 — guaranteed by max-normalization but not by mean-normalization
(where off-resonance ≈ 1 + contrast×dip_fraction, typically 1.03–1.10).

This script:
1. Generates synthetic ESR14N mean-normalized ODMR data with known parameters.
2. Computes initial guesses with both baseline approaches.
3. Evaluates chi² = mean((data - model(initial_params))²) per pixel for each approach.
4. Reports summary statistics showing the improvement.

No GPU required — only the initial-parameter quality is assessed (pre-fit residual).

Usage
-----
    uv run python scripts/compare_guesser_chi2.py [--n-pixels N] [--snr SNR] [--seed S]
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

# ── model function (same as esr14n in models.py) ─────────────────────────────

AHYP_14N = 0.002158  # GHz


def esr14n_model(freq: NDArray, params: NDArray) -> NDArray:
    """ESR14N Lorentzian triplet. params shape: (n_pixel, 6)."""
    params = np.atleast_2d(params)
    center = params[:, 0:1]
    width_sq = params[:, 1:2] ** 2
    c0, c1, c2 = params[:, 2:3], params[:, 3:4], params[:, 4:5]
    offset = params[:, 5:6]
    dip1 = c0 * width_sq / ((freq - center + AHYP_14N) ** 2 + width_sq)
    dip2 = c1 * width_sq / ((freq - center) ** 2 + width_sq)
    dip3 = c2 * width_sq / ((freq - center - AHYP_14N) ** 2 + width_sq)
    return 1 + offset - dip1 - dip2 - dip3


def chi2_per_pixel(data: NDArray, model_vals: NDArray) -> NDArray:
    """Mean squared residual per pixel. Both shapes: (n_pixel, n_freq)."""
    return np.mean((data - model_vals) ** 2, axis=-1)


# ── baseline estimators ───────────────────────────────────────────────────────


def baseline_fixed(pixel: NDArray) -> float:
    """Old approach: hardcoded 1.0."""
    return 1.0


def baseline_edge(pixel: NDArray, frac: float = 0.10) -> float:
    """New approach: mean of first and last `frac` fraction of points."""
    n = len(pixel)
    n_edge = max(1, int(n * frac))
    return float((np.mean(pixel[:n_edge]) + np.mean(pixel[n - n_edge :])) / 2)


# ── per-pixel guesser (pure Python, matches numba logic) ─────────────────────


def _normalize_pixel(pixel: NDArray, bl: float) -> NDArray:
    """Cumulative-sum normalization with given baseline."""
    cs = np.cumsum(pixel - bl)
    cs -= cs.min()
    mx = cs.max()
    return cs / mx if mx > 0 else cs


def guess_center(pixel: NDArray, freq: NDArray, bl: float) -> float:
    norm = _normalize_pixel(pixel, bl)
    return float(freq[np.argmin(np.abs(norm - 0.5))])


def guess_width(
    pixel: NDArray, freq: NDArray, bl: float, vmin: float = 0.35, vmax: float = 0.65
) -> float:
    norm = _normalize_pixel(pixel, bl)
    lidx = int(np.argmin(np.abs(norm - vmin)))
    ridx = int(np.argmin(np.abs(norm - vmax)))
    return abs(float(freq[ridx]) - float(freq[lidx]))


def guess_contrast(pixel: NDArray) -> float:
    mx = np.nanmax(pixel)
    mn = np.nanmin(pixel)
    return float(abs((mx - mn) / mx)) if mx != 0 else 0.0


def build_initial_params(data: NDArray, freq: NDArray, baseline_fn: callable) -> NDArray:
    """Build (n_pixel, 6) initial param array using the given baseline estimator."""
    n_pixel = data.shape[0]
    params = np.zeros((n_pixel, 6), dtype=np.float32)
    for px in range(n_pixel):
        pixel = data[px]
        bl = baseline_fn(pixel)
        contrast = guess_contrast(pixel)
        params[px, 0] = guess_center(pixel, freq, bl)  # center
        params[px, 1] = guess_width(pixel, freq, bl)  # width
        params[px, 2] = contrast / 3  # contrast_0
        params[px, 3] = contrast / 3  # contrast_1
        params[px, 4] = contrast / 3  # contrast_2
        params[px, 5] = 0.0  # offset
    return params


# ── synthetic data generation ─────────────────────────────────────────────────


def make_synthetic_data(
    n_pixel: int,
    freq: NDArray,
    rng: np.random.Generator,
    snr: float = 50.0,
) -> tuple[NDArray, NDArray]:
    """Generate mean-normalized ESR14N spectra with random parameters.

    Returns:
    -------
    data : (n_pixel, n_freq) — mean-normalized spectra
    true_params : (n_pixel, 6) — ground-truth parameters
    """
    n_freq = len(freq)
    f_min, f_max = freq.min(), freq.max()
    f_mid = (f_min + f_max) / 2
    f_span = f_max - f_min

    # Randomise center (±30% of half-span from midpoint, but keep dips inside frange)
    margin = AHYP_14N * 2 + 0.003
    center = rng.uniform(f_mid - f_span * 0.25, f_mid + f_span * 0.25, n_pixel)
    center = np.clip(center, f_min + margin, f_max - margin)

    width = rng.uniform(0.001, 0.003, n_pixel)
    contrast = rng.uniform(0.04, 0.12, n_pixel)

    true_params = np.column_stack(
        [
            center,
            width,
            contrast / 3,
            contrast / 3,
            contrast / 3,
            np.zeros(n_pixel),
        ]
    )

    # Evaluate noiseless spectra (shape: n_pixel × n_freq)
    clean = esr14n_model(freq[np.newaxis, :], true_params)  # (n_pixel, n_freq)

    # Add Gaussian noise
    noise_std = contrast.mean() / snr
    noisy = clean + rng.normal(0, noise_std, (n_pixel, n_freq))

    # Mean-normalize: divide each pixel by its mean across frequencies
    # After this, off-resonance ≈ 1 + contrast_fraction (typically 1.03–1.10)
    pixel_means = noisy.mean(axis=1, keepdims=True)
    data = noisy / pixel_means

    return data.astype(np.float32), true_params.astype(np.float32)


# ── main ─────────────────────────────────────────────────────────────────────


def _run(n_pixel: int, snr: float, seed: int) -> None:
    rng = np.random.default_rng(seed)

    # Frequency axis: typical low frange
    freq = np.linspace(2.820, 2.870, 50, dtype=np.float32)

    print(f"Synthetic ESR14N: {n_pixel} pixels, SNR={snr:.0f}, seed={seed}")
    print(f"Frequency: {freq[0]:.3f}–{freq[-1]:.3f} GHz  ({len(freq)} points)")
    print()

    data, true_params = make_synthetic_data(n_pixel, freq, rng, snr=snr)

    # Sanity: show off-resonance baseline distribution (should be ~1.03–1.10)
    edge = max(1, len(freq) // 10)
    off_res = (data[:, :edge].mean(axis=1) + data[:, -edge:].mean(axis=1)) / 2
    print(
        f"Off-resonance baseline: mean={off_res.mean():.4f}  "
        f"std={off_res.std():.4f}  "
        f"[{off_res.min():.4f}, {off_res.max():.4f}]"
    )
    print()

    # ── old guesser (baseline = 1.0) ─────────────────────────────────────────
    params_old = build_initial_params(data, freq, baseline_fixed)
    model_old = esr14n_model(freq[np.newaxis, :], params_old)
    chi2_old = chi2_per_pixel(data, model_old)

    # ── new guesser (edge-based baseline) ────────────────────────────────────
    params_new = build_initial_params(data, freq, baseline_edge)
    model_new = esr14n_model(freq[np.newaxis, :], params_new)
    chi2_new = chi2_per_pixel(data, model_new)

    # ── oracle: true parameters ───────────────────────────────────────────────
    model_true = esr14n_model(freq[np.newaxis, :], true_params)
    chi2_true = chi2_per_pixel(data, model_true)

    # ── center accuracy ───────────────────────────────────────────────────────
    center_err_old = np.abs(params_old[:, 0] - true_params[:, 0]) * 1000  # MHz
    center_err_new = np.abs(params_new[:, 0] - true_params[:, 0]) * 1000  # MHz

    # ── report ────────────────────────────────────────────────────────────────
    def row(label: str, arr: NDArray) -> str:
        return (
            f"  {label:<26}  median={np.median(arr):.2e}  "
            f"mean={np.mean(arr):.2e}  "
            f"p99={np.percentile(arr, 99):.2e}"
        )

    print("Initial chi² (mean squared residual per pixel)")
    print("─" * 70)
    print(row("old (baseline = 1.0)", chi2_old))
    print(row("new (edge-based)    ", chi2_new))
    print(row("oracle (true params)", chi2_true))
    print()

    improvement = (chi2_old - chi2_new) / (chi2_old + 1e-30) * 100
    print(
        f"  chi² improvement: median {np.median(improvement):.1f}%  "
        f"mean {np.mean(improvement):.1f}%"
    )
    print()

    print("Center estimate error (MHz)")
    print("─" * 70)
    print(row("old (baseline = 1.0)", center_err_old))
    print(row("new (edge-based)    ", center_err_new))
    print()

    # fraction of pixels where new is better
    frac_better_chi2 = np.mean(chi2_new < chi2_old) * 100
    frac_better_center = np.mean(center_err_new < center_err_old) * 100
    print(
        f"  Fraction of pixels where new < old:  "
        f"chi²={frac_better_chi2:.1f}%  center_err={frac_better_center:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-pixels", type=int, default=2000, help="Number of synthetic pixels (default: 2000)"
    )
    parser.add_argument(
        "--snr", type=float, default=30.0, help="Signal-to-noise ratio (default: 30)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    _run(args.n_pixels, args.snr, args.seed)


if __name__ == "__main__":
    main()
