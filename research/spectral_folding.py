# ruff: noqa
"""Spectral folding prototype for ODMR resonance detection.

Idea borrowed from Mössbauer spectroscopy: fold the *mean* spectrum around a
candidate centre frequency and minimise the residual between the two halves.
The optimal fold point gives the resonance centre without parametric fitting.

Because we fold the spatially-averaged spectrum (not each pixel), the SNR is
~sqrt(N_pixels) × better than per-pixel, and the whole computation is O(1) in
image size.  The fold centre can then serve as a robust initialisation for
per-pixel dip search or fitting.

Usage (from repo root):
    uv run python research/spectral_folding.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from qdmpy_core.constants import GAMMA_NV
from qdmpy_core.measurement import Measurement
from qdmpy_core.odmr.analysis import b111_from_dip_positions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("tests/data/MIL2_FOV1")
BIN_FACTOR = 4


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def fold_residuals(
    spectrum: NDArray,
    freqs: NDArray,
    n_candidates: int = 500,
) -> tuple[NDArray, NDArray]:
    """Sweep candidate centres and return residual for each.

    Args:
        spectrum: 1-D intensity array (n_freq,).
        freqs: 1-D frequency array in GHz, monotonically increasing.
        n_candidates: Number of centre points to evaluate.

    Returns:
        (centers, residuals) — both shape (n_candidates,).
    """
    margin = 0.2 * (freqs[-1] - freqs[0])
    centers = np.linspace(freqs[0] + margin, freqs[-1] - margin, n_candidates)

    # Pre-compute step size for offset grid
    step = float(np.median(np.diff(freqs)))
    residuals = np.empty(n_candidates)

    for i, c in enumerate(centers):
        reach = min(c - freqs[0], freqs[-1] - c)
        offsets = np.arange(step, reach + step / 2, step)
        if len(offsets) == 0:
            residuals[i] = np.inf
            continue
        left = np.interp(c - offsets, freqs, spectrum)
        right = np.interp(c + offsets, freqs, spectrum)
        residuals[i] = float(np.sum((left - right) ** 2))

    return centers, residuals


def find_fold_center(
    spectrum: NDArray,
    freqs: NDArray,
    n_candidates: int = 500,
) -> tuple[float, NDArray, NDArray]:
    """Find the fold centre that best symmetrises the spectrum.

    Returns:
        (best_center_ghz, centers, residuals)
    """
    centers, residuals = fold_residuals(spectrum, freqs, n_candidates)
    best = centers[int(np.argmin(residuals))]
    return best, centers, residuals


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_single_band(
    mean_spec: NDArray,
    freqs: NDArray,
    fold_center: float,
    centers: NDArray,
    residuals: NDArray,
    label: str,
) -> None:
    """Two-panel plot: spectrum + residual landscape for one band."""
    fig, (ax_spec, ax_res) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    fig.suptitle(f"Spectral folding — {label}")

    # --- spectrum panel ---
    ax_spec.plot(freqs, mean_spec, "k-", lw=1.4, label="mean spectrum")
    ax_spec.axvline(
        fold_center,
        color="tab:red",
        ls="--",
        lw=1.5,
        label=f"fold centre = {fold_center:.5f} GHz",
    )
    ax_spec.axvline(
        freqs[np.argmin(mean_spec)],
        color="tab:blue",
        ls=":",
        lw=1.5,
        label=f"argmin dip = {freqs[np.argmin(mean_spec)]:.5f} GHz",
    )
    ax_spec.set_ylabel("Normalised intensity")
    ax_spec.set_xlabel("Frequency (GHz)")
    ax_spec.legend(fontsize=8)

    # mirrored half overlay
    fc = fold_center
    reach = min(fc - freqs[0], freqs[-1] - fc)
    f_left = np.linspace(fc - reach, fc, 200)
    f_right = 2 * fc - f_left[::-1]
    spec_left = np.interp(f_left, freqs, mean_spec)
    spec_right = np.interp(f_right, freqs, mean_spec)
    ax_spec.plot(f_left, spec_right[::-1], "tab:red", lw=0.8, alpha=0.5, label="right half mirrored")
    ax_spec.legend(fontsize=8)

    # --- residual panel ---
    ax_res.plot(centers, residuals, "b-", lw=1)
    ax_res.axvline(fold_center, color="tab:red", ls="--", lw=1.5)
    ax_res.set_ylabel("Fold residual (a.u.)")
    ax_res.set_xlabel("Candidate centre (GHz)")

    fig.tight_layout()


def plot_all_bands(
    results: dict[str, dict],
    freq_ghz_arr: NDArray,
    frange_labels: list[str],
) -> None:
    """4-panel overview: one row per polarity × one column per freq_range."""
    pols = list(results.keys())
    n_pol = len(pols)
    n_frange = len(frange_labels)

    fig, axes = plt.subplots(n_pol, n_frange, figsize=(5 * n_frange, 4 * n_pol), squeeze=False)
    fig.suptitle("Mean-spectrum fold centres (all bands)", fontsize=12)

    for pi, pol in enumerate(pols):
        for fi, frange in enumerate(frange_labels):
            ax = axes[pi, fi]
            res = results[pol][frange]
            freqs = freq_ghz_arr[fi]
            spec = res["mean_spec"]
            fc = res["fold_center"]

            ax.plot(freqs, spec, "k-", lw=1.2)
            ax.axvline(fc, color="tab:red", ls="--", lw=1.5,
                       label=f"fold {fc:.5f}")
            ax.axvline(freqs[np.argmin(spec)], color="tab:blue", ls=":",
                       label=f"argmin {freqs[np.argmin(spec)]:.5f}")
            ax.set_title(f"pol={pol}  frange={frange}", fontsize=9)
            ax.set_xlabel("Frequency (GHz)", fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.legend(fontsize=7)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading data from {DATA_DIR} (bin={BIN_FACTOR}) …")
    m = Measurement.from_folder(DATA_DIR, bin_factor=BIN_FACTOR, normalize=True)
    data = m.odmr.processed_data.data

    freq_ghz_arr = data.coords["freq_ghz"].values   # (n_frange, n_freq)
    frange_labels = list(data.coords["freq_range"].values)
    pol_labels = list(data.coords["polarity"].values)
    pixel_spacing_um = m.pixel_spacing * 1e6

    print(f"  shape:  {data.shape}")
    print(f"  frange: {frange_labels},  pol: {pol_labels}")

    # --- Fold the mean spectrum per (polarity, freq_range) ---
    results: dict[str, dict] = {}
    print()
    for pol in pol_labels:
        results[pol] = {}
        for fi, frange in enumerate(frange_labels):
            freqs = freq_ghz_arr[fi]
            # Average over all spatial pixels → shape (n_freq,)
            mean_spec = data.sel(polarity=pol, freq_range=frange).values.mean(axis=(0, 1))

            fold_center, centers, residuals = find_fold_center(mean_spec, freqs)
            argmin_dip = freqs[np.argmin(mean_spec)]

            results[pol][frange] = {
                "mean_spec": mean_spec,
                "fold_center": fold_center,
                "argmin_dip": argmin_dip,
                "centers": centers,
                "residuals": residuals,
            }

            print(
                f"  pol={pol}  frange={frange}:  "
                f"fold={fold_center:.6f} GHz   "
                f"argmin={argmin_dip:.6f} GHz   "
                f"Δ={1000*(fold_center - argmin_dip):.2f} MHz"
            )

    # --- Derive B₁₁₁ estimate from mean-fold centres ---
    # δB[pol] = sign[pol] × (f_high − f_low) / 2 / GAMMA_NV  [µT]
    print()
    sign = {"neg": -1.0, "pos": 1.0}
    delta_fold = {}
    delta_argmin = {}
    for pol in pol_labels:
        f_low_fold = results[pol]["low"]["fold_center"]
        f_high_fold = results[pol]["high"]["fold_center"]
        delta_fold[pol] = sign[pol] * (f_high_fold - f_low_fold) / 2.0 / GAMMA_NV * 1e6

        f_low_am = results[pol]["low"]["argmin_dip"]
        f_high_am = results[pol]["high"]["argmin_dip"]
        delta_argmin[pol] = sign[pol] * (f_high_am - f_low_am) / 2.0 / GAMMA_NV * 1e6

        print(
            f"  pol={pol}:  δB(fold)={delta_fold[pol]:.2f} µT   "
            f"δB(argmin)={delta_argmin[pol]:.2f} µT"
        )

    b111_rem_fold = (delta_fold["neg"] + delta_fold["pos"]) / 2
    b111_ind_fold = (delta_fold["neg"] - delta_fold["pos"]) / 2
    b111_rem_am = (delta_argmin["neg"] + delta_argmin["pos"]) / 2
    b111_ind_am = (delta_argmin["neg"] - delta_argmin["pos"]) / 2

    print(f"\n  mean-field estimate (fold)  :  B111_rem={b111_rem_fold:.2f} µT   B111_ind={b111_ind_fold:.2f} µT")
    print(f"  mean-field estimate (argmin):  B111_rem={b111_rem_am:.2f} µT   B111_ind={b111_ind_am:.2f} µT")

    # Also compare to per-pixel argmin map statistics for sanity
    print("\nPer-pixel argmin map (for comparison):")
    b111_map = b111_from_dip_positions(data)
    for key in ("remanent", "induced"):
        arr = b111_map[key]
        print(f"  {key}: mean={arr.mean():.2f}  median={np.median(arr):.2f}  std={arr.std():.2f} µT")

    # --- Plots ---
    # Detailed single-band diagnostic for neg/low
    r = results["neg"]["low"]
    plot_single_band(
        r["mean_spec"], freq_ghz_arr[0], r["fold_center"],
        r["centers"], r["residuals"], "pol=neg  frange=low"
    )

    # Overview of all 4 bands
    plot_all_bands(results, freq_ghz_arr, frange_labels)

    out_dir = Path("research/output")
    out_dir.mkdir(exist_ok=True)
    for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
        path = out_dir / f"spectral_folding_{i}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")

    plt.show()


if __name__ == "__main__":
    main()
