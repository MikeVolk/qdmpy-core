"""Plot old vs new initial-parameter guesses for a few sample pixels.

Produces a grid of subplots: one row per pixel, two columns (low frange,
high frange). Each panel shows the raw data, old guess overlay, and new
guess overlay so the improvement is visually obvious.

Usage
-----
    uv run python scripts/plot_guess_comparison.py [--data tests/data/MIL2_FOV1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.constants import DEFAULT_VMAX, DEFAULT_VMIN
from qdmpy.fitting.guess import (
    cumsum_center,
    cumsum_contrast,
    cumsum_width,
    get_model_by_peaks,
    guess_model,
)
from qdmpy.fitting.guesser import ParameterGuesser


def load_data(data_folder: str, bin_factor: int = 4) -> tuple[NDArray, NDArray, int]:
    """Load and preprocess ODMR data."""
    logger.disable("qdmpy")
    from qdmpy.odmr.data import ODMRData
    from qdmpy.odmr.io import MatlabLoader
    from qdmpy.odmr.manager import ODMR
    from qdmpy.odmr.processors import BinningProcessor, NormalizationProcessor

    loader = MatlabLoader(data_folder=data_folder)
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    if bin_factor > 1:
        odmr.processor_manager.add_processor(BinningProcessor(bin_factor=bin_factor))
    odmr.processor_manager.add_processor(NormalizationProcessor())
    odmr.process_data()

    processed = odmr.processed_data
    raw = processed.data.values
    n_pol, n_frange, h, w, n_freq = raw.shape
    flat_data = raw.reshape(n_pol, n_frange, h * w, n_freq)
    freq = processed.frequencies

    detected = guess_model(flat_data)
    return flat_data, freq, detected.n_peaks


def build_old_params(
    flat_data: NDArray,
    freq: NDArray,
    model: object,
    n_peaks: int,
) -> NDArray:
    """Reproduce the old guesser logic (before this PR)."""
    n_pol, n_frange, n_pixel, _ = flat_data.shape
    params = np.zeros((n_pol, n_frange, n_pixel, model.n_parameters), dtype=np.float32)
    for idx, pname in enumerate(model.parameter_names):
        ptype = model.parameter_types[pname]
        if ptype == "center":
            params[:, :, :, idx] = cumsum_center(flat_data, freq)
        elif ptype == "contrast":
            params[:, :, :, idx] = cumsum_contrast(flat_data)  # no /n_peaks
        elif ptype == "width":
            if n_peaks == 3:
                vmin, vmax = 0.35, 0.65
            elif n_peaks == 2:
                vmin, vmax = 0.4, 0.6
            else:
                vmin, vmax = DEFAULT_VMIN, DEFAULT_VMAX
            params[:, :, :, idx] = cumsum_width(flat_data, freq, vmin, vmax)
        elif ptype == "offset":
            n_f = flat_data.shape[-1]
            n_edge = max(1, n_f // 10)
            baseline = (
                np.mean(flat_data[..., :n_edge], axis=-1)
                + np.mean(flat_data[..., -n_edge:], axis=-1)
            ) / 2.0
            params[:, :, :, idx] = (baseline - 1.0).astype(np.float32)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent.parent / "tests" / "data" / "MIL2_FOV1"),
    )
    parser.add_argument("--bin", type=int, default=4)
    parser.add_argument("--n-pixels", type=int, default=4)
    parser.add_argument("--pol", type=int, default=0, help="Polarity index to plot")
    parser.add_argument("-o", "--output", default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    flat_data, freq, n_peaks = load_data(args.data, args.bin)
    _n_pol, n_frange, _n_pixel, _n_freq = flat_data.shape
    model = get_model_by_peaks(n_peaks)
    print(f"Model: {model.name} ({n_peaks} peaks), data: {flat_data.shape}")

    # New params (current code)
    guesser = ParameterGuesser(model, freq)
    params_new = guesser.guess(flat_data)

    # Old params (pre-PR logic)
    params_old = build_old_params(flat_data, freq, model, n_peaks)

    # Pick sample pixels spread across the FOV
    rng = np.random.default_rng(42)
    # Choose pixels with decent contrast (not dead pixels)
    contrast = cumsum_contrast(flat_data)
    median_contrast = np.median(contrast[args.pol, 0])
    good_mask = contrast[args.pol, 0] > median_contrast * 0.5
    good_indices = np.where(good_mask)[0]
    sample_px = rng.choice(good_indices, size=min(args.n_pixels, len(good_indices)), replace=False)
    sample_px.sort()

    pol = args.pol
    fig, axes = plt.subplots(
        len(sample_px),
        n_frange,
        figsize=(6 * n_frange, 3 * len(sample_px)),
        squeeze=False,
        sharex="col",
    )

    frange_labels = ["low frange", "high frange"]

    for row, px in enumerate(sample_px):
        for col in range(n_frange):
            ax = axes[row, col]
            f = freq[col] * 1e3  # GHz -> MHz for readability

            # Raw data
            ax.plot(f, flat_data[pol, col, px], "k.", ms=3, alpha=0.6, label="data")

            # Old guess
            old_curve = model.func(freq[col], params_old[pol, col, px : px + 1])
            ax.plot(f, old_curve[0], "r-", lw=1.5, alpha=0.8, label="old guess")

            # New guess
            new_curve = model.func(freq[col], params_new[pol, col, px : px + 1])
            ax.plot(f, new_curve[0], "b-", lw=1.5, alpha=0.8, label="new guess")

            # Residual annotation
            old_mse = float(np.mean((flat_data[pol, col, px] - old_curve[0]) ** 2))
            new_mse = float(np.mean((flat_data[pol, col, px] - new_curve[0]) ** 2))
            ax.text(
                0.02,
                0.02,
                f"MSE old={old_mse:.1e}\n    new={new_mse:.1e}",
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                fontfamily="monospace",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.7},
            )

            if row == 0:
                ax.set_title(frange_labels[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"pixel {px}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")
            if row == len(sample_px) - 1:
                ax.set_xlabel("Frequency (MHz)")

    fig.suptitle(
        f"Initial guess comparison — {model.name}, pol={pol}\n"
        f"{Path(args.data).name} (bin={args.bin})",
        fontsize=13,
    )
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
