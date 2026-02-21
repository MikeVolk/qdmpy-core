"""Visualization module for QDMpy.

This module provides plotting functions for visualizing data from Quantum Diamond
Microscopy (QDM) measurements, including magnetic field maps and spatial parameter maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

from QDMpy.utils import double_norm

if TYPE_CHECKING:
    from QDMpy.fitting.result import FitResult

# Set white background for all QDMpy figures
mpl.rcParams["figure.facecolor"] = "white"


def plot_fit_result_field_map(
    result: FitResult, save: bool = False, filename: str | None = None
) -> None:
    """Plot magnetic field map from FitResult.

    Args:
        result: FitResult object containing fitted parameters
        save: Whether to save the plot to file
        filename: Custom filename for saving (optional)
    """
    b_field = result.calculate_b_field()

    title = f"Magnetic Field Map ({result.model_name})"
    cmap = "viridis"
    colorbar_label = "Magnetic Field (T)"

    _fig, ax = plt.subplots(figsize=(8, 6))

    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, 0, height * pixel_spacing_um)

    im = ax.imshow(
        b_field,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title(title)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"b_field_map_{result.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_fit_result_parameter_map(
    result: FitResult,
    param_name: str,
    save: bool = False,
    filename: str | None = None,
) -> None:
    """Plot spatial map of fitted parameter from FitResult.

    Args:
        result: FitResult object containing fitted parameters
        param_name: Name of parameter to plot (e.g., 'center', 'width_0', 'contrast')
        save: Whether to save the plot to file
        filename: Custom filename for saving (optional)
    """
    param_map = result.get_parameter_map(param_name)

    param_labels = {
        "center": "Resonance Center (Hz)",
        "width_0": "Linewidth (Hz)",
        "width_1": "Linewidth 1 (Hz)",
        "width_2": "Linewidth 2 (Hz)",
        "contrast": "ODMR Contrast",
        "offset": "Baseline Offset",
        "chi2": "Fit Quality (χ²)",
        "states": "Fit State",
    }

    title = f"{param_name.replace('_', ' ').title()} Map ({result.model_name})"
    colorbar_label = param_labels.get(param_name, param_name.title())
    cmap = "viridis"

    _fig, ax = plt.subplots(figsize=(8, 6))

    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, 0, height * pixel_spacing_um)

    im = ax.imshow(
        param_map,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title(title)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"{param_name}_map_{result.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_fit_result_overview(
    result: FitResult, save: bool = False, filename: str | None = None
) -> None:
    """Plot overview of fit results with multiple parameter maps.

    Args:
        result: FitResult object containing fitted parameters
        save: Whether to save the plot to file
        filename: Custom filename for saving (optional)
    """
    plot_params = ["center", "width_0", "contrast", "chi2"]
    available_params = [p for p in plot_params if p in result.parameters]

    b_field = result.calculate_b_field()

    n_plots = len(available_params) + 1  # +1 for B-field
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    _fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = [axes] if nrows == 1 and ncols == 1 else axes.flatten()

    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, 0, height * pixel_spacing_um)

    plot_idx = 0

    ax = axes[plot_idx]
    im = ax.imshow(b_field, extent=extent, origin="lower", cmap="viridis", aspect="equal")
    ax.set_title("Magnetic Field (T)")
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    plt.colorbar(im, ax=ax)
    plot_idx += 1

    for param in available_params:
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        param_map = result.get_parameter_map(param)

        im = ax.imshow(param_map, extent=extent, origin="lower", cmap="viridis", aspect="equal")
        ax.set_title(f"{param.replace('_', ' ').title()}")
        ax.set_xlabel("x [μm]")
        ax.set_ylabel("y [μm]")
        plt.colorbar(im, ax=ax)
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Fit Results Overview ({result.model_name})", fontsize=14)
    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"fit_overview_{result.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


__all__ = [
    "double_norm",
    "plot_fit_result_field_map",
    "plot_fit_result_overview",
    "plot_fit_result_parameter_map",
]
