"""Benchmark initial-parameter guess quality: old (cumsum) vs new (halfpower + contrast fix).

Loads real ODMR data, runs both the old and new guesser logic, evaluates the
model at the initial parameters, and reports per-pixel residual statistics.

The new guesses must produce equal or lower residuals to validate the changes.

Usage
-----
    # Capture baseline (before code changes)
    uv run python scripts/benchmark_guesses.py --save-baseline

    # Compare against saved baseline (after code changes)
    uv run python scripts/benchmark_guesses.py

    # Use custom data folder
    uv run python scripts/benchmark_guesses.py --data tests/data/FOV18x
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BASELINE_PATH = Path(__file__).parent / "benchmark_baseline.json"


def chi2_per_pixel(data: NDArray, model_vals: NDArray) -> NDArray:
    """Mean squared residual per pixel. Both shapes: (n_pixel, n_freq)."""
    return np.mean((data - model_vals) ** 2, axis=-1)


def load_odmr_data(data_folder: str, bin_factor: int = 2) -> tuple[NDArray, NDArray, int]:
    """Load and preprocess ODMR data, returning flat data, freq, and n_peaks.

    Returns:
        flat_data: (n_pol, n_frange, n_pixel, n_freq)
        freq: (n_frange, n_freq) in GHz
        n_peaks: detected number of peaks
    """
    from loguru import logger

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
    raw = processed.data.values  # (pol, frange, y, x, freq)
    n_pol, n_frange, h, w, n_freq = raw.shape
    flat_data = raw.reshape(n_pol, n_frange, h * w, n_freq)
    freq = processed.frequencies  # (n_frange, n_freq)

    from qdmpy.fitting.guess import guess_model

    detected = guess_model(flat_data)
    return flat_data, freq, detected.n_peaks


def evaluate_guesses(flat_data: NDArray, freq: NDArray, n_peaks: int) -> dict[str, float]:
    """Run the current guesser and evaluate guess quality via residuals.

    Returns:
        Dict with mean, median, p95, max residual, and timing.
    """
    from qdmpy.fitting.guess import get_model_by_peaks
    from qdmpy.fitting.guesser import ParameterGuesser

    model = get_model_by_peaks(n_peaks)
    guesser = ParameterGuesser(model, freq)

    # Warmup (JIT compilation)
    _ = guesser.guess(flat_data)
    guesser.reset()

    # Timed run (average of 3)
    times = []
    for _ in range(3):
        guesser.reset()
        t0 = time.perf_counter()
        params = guesser.guess(flat_data)
        times.append(time.perf_counter() - t0)

    # Evaluate model at guessed parameters for every pixel
    n_pol, n_frange, _n_pixel, _n_freq_pts = flat_data.shape
    all_residuals = []
    for p in range(n_pol):
        for r in range(n_frange):
            pixel_params = params[p, r]  # (n_pixel, n_params)
            model_vals = model.func(freq[r], pixel_params)  # (n_pixel, n_freq)
            residuals = chi2_per_pixel(flat_data[p, r], model_vals)
            all_residuals.append(residuals)

    all_residuals_arr = np.concatenate(all_residuals)

    return {
        "mean_residual": float(np.mean(all_residuals_arr)),
        "median_residual": float(np.median(all_residuals_arr)),
        "p95_residual": float(np.percentile(all_residuals_arr, 95)),
        "max_residual": float(np.max(all_residuals_arr)),
        "mean_time_ms": float(np.mean(times) * 1e3),
        "n_pixels_total": len(all_residuals_arr),
    }


def print_results(label: str, results: dict[str, float]) -> None:
    """Print formatted results."""
    print(f"  {label}:")
    print(f"    mean_residual  = {results['mean_residual']:.2e}")
    print(f"    median_residual= {results['median_residual']:.2e}")
    print(f"    p95_residual   = {results['p95_residual']:.2e}")
    print(f"    max_residual   = {results['max_residual']:.2e}")
    print(f"    guess_time     = {results['mean_time_ms']:.1f} ms")
    print(f"    n_pixels       = {results['n_pixels_total']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent.parent / "tests" / "data" / "FOV18x"),
        help="Path to ODMR data folder",
    )
    parser.add_argument("--bin", type=int, default=4, metavar="FACTOR")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as baseline for future comparison",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data} (bin={args.bin})...")
    flat_data, freq, n_peaks = load_odmr_data(args.data, args.bin)
    print(f"Data shape: {flat_data.shape}, freq shape: {freq.shape}, n_peaks: {n_peaks}")
    print()

    results = evaluate_guesses(flat_data, freq, n_peaks)

    print("Current guesser results")
    print("-" * 50)
    print_results("current", results)

    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(results, indent=2))
        print(f"Baseline saved to {BASELINE_PATH}")
        return

    # Compare against baseline if it exists
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())
        print("Baseline (saved previously)")
        print("-" * 50)
        print_results("baseline", baseline)

        print("Comparison")
        print("-" * 50)
        for key in ("mean_residual", "median_residual", "p95_residual"):
            old = baseline[key]
            new = results[key]
            pct = (old - new) / (old + 1e-30) * 100
            direction = "BETTER" if new <= old else "WORSE"
            print(f"  {key}: {old:.2e} -> {new:.2e}  ({pct:+.1f}% {direction})")

        time_old = baseline.get("mean_time_ms", 0)
        time_new = results["mean_time_ms"]
        print(f"  timing: {time_old:.1f} ms -> {time_new:.1f} ms")
    else:
        print(
            f"No baseline found at {BASELINE_PATH}. "
            f"Run with --save-baseline first to capture a baseline."
        )


if __name__ == "__main__":
    main()
