#!/usr/bin/env python3
"""Sample script for loading, binning, and fitting 15N diamond ODMR data.

Demonstrates the manual mid-level pipeline (loader -> ODMR -> processors ->
FitManager) rather than the qdmpy.load() one-liner shown in example_data.py
-- useful when you need direct control over the processing pipeline.

Usage:
    python fit_15n_sample.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from qdmpy import ODMR, BinningProcessor, FitManager, MatlabLoader, NormalizationProcessor, ODMRData

# tests/data/ is gitignored fixture data present in a dev checkout; point
# this at your own data folder if you don't have it.
DATA_FOLDER = Path(__file__).resolve().parents[1] / "tests" / "data" / "FOV18x"


def main() -> None:
    """Load, bin, fit, and visualize 15N ODMR data."""
    loader = MatlabLoader(data_folder=str(DATA_FOLDER))
    raw = ODMRData.from_loader(loader)

    odmr = ODMR(raw)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.processor_manager.add_processor(NormalizationProcessor())
    odmr.process_data()

    logger.info("Fitting ESR15N model")
    fit_manager = FitManager("ESR15N", backend="scipy")
    fit_result = fit_manager.fit(
        odmr.processed_data.data, odmr.processed_data.frequencies, pixel_spacing=4e-6
    )

    quality = fit_result.metadata["quality_metrics"]
    logger.info(
        "Fit complete: mean_chi2={:.4f}, convergence_rate={:.1%}",
        quality["mean_chi2"],
        quality["convergence_rate"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    h, w = odmr.processed_data.shape[2], odmr.processed_data.shape[3]
    pixel_coords = [(y, x) for y in (h // 4, 3 * h // 4) for x in (w // 4, 3 * w // 4)]

    for ax, (y, x) in zip(axes.flat, pixel_coords, strict=True):
        freq, spec = odmr.spectrum(y=y, x=x, polarity="neg", freq_range="low", processed=True)
        ax.plot(freq, spec, "b-", linewidth=2)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("ODMR Signal (normalized)")
        ax.set_title(f"Pixel ({y}, {x}) ODMR Spectrum")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(__file__).parent / "odmr_spectra_sample.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved sample spectra to {}", output_path)
    plt.show()


if __name__ == "__main__":
    main()
