#!/usr/bin/env python3
"""Example script for loading, fitting, and visualizing ODMR data with QDMpy.

Demonstrates the one-line entry point for the common workflow: load a folder
of MATLAB .mat files (plus LED/laser images), apply standard processing, and
fit ODMR spectra to a model.

Usage:
    python example_data.py

Expected folder structure:
    /path/to/data/
    ├── laser.csv
    ├── laser.jpg
    ├── LED.csv
    ├── LED.jpg
    ├── run_00000.mat
    └── run_00001.mat
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

import qdmpy

# tests/data/ is gitignored fixture data present in a dev checkout; point
# this at your own data folder if you don't have it.
DATA_FOLDER = Path(__file__).resolve().parents[1] / "tests" / "data" / "FOV18x"


def main() -> None:
    """Load, fit, and visualize an ODMR measurement."""
    logger.info("Loading measurement from {}", DATA_FOLDER)
    measurement = qdmpy.load(DATA_FOLDER, bin_factor=2)

    logger.info("Fitting ODMR spectra")
    result = measurement.fit_odmr(backend="scipy")

    b111 = result.b111_remanent
    logger.info("b111_remanent shape={}, mean={:.2f} uT", b111.shape, b111.mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(b111, cmap="RdBu_r")
    ax.set_title("B111 remanent field")
    fig.colorbar(im, ax=ax, label="uT")

    output_path = Path(__file__).parent / "b111_remanent.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved B111 map to {}", output_path)
    plt.show()


if __name__ == "__main__":
    main()
