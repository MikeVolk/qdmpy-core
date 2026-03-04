"""Fit-result visualizations: field maps, parameter maps, and overviews."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from qdmpy.exceptions import DataShapeError
from qdmpy.plotting._common import _add_colorbar, _label_spatial_axes

if TYPE_CHECKING:
    from qdmpy.fitting.result import FitResult


def plot_fit_result_field_map(
    result: FitResult, save: bool = False, filename: str | None = None
) -> None:
    """Plot magnetic field map from FitResult.

    Uses B111 remanent (uT, diverging colormap) for multi-range models such as
    ESR14N/ESR15N.  Falls back to the legacy ``calculate_b_field()`` (T) for
    single-range models where B111 decomposition is not available.

    Args:
        result: FitResult or QDMResult containing fitted parameters.
        save: Whether to save the plot to file.
        filename: Custom filename for saving (optional).
    """
    logger.debug("Plotting field map (model={})", result.model_name)
    try:
        b_field = result.b111_remanent  # (H, W), uT
        colorbar_label = "B111 remanent (µT)"
        cmap = "RdBu_r"
        vmax: float | None = float(np.nanpercentile(np.abs(b_field), 99))
        vmin: float | None = -vmax
    except (DataShapeError, AttributeError, TypeError):
        logger.warning("B111 remanent not available, falling back to legacy B-field calculation")
        b_field = result.calculate_b_field()  # (H, W), T
        colorbar_label = "Magnetic Field (T)"
        cmap = "viridis"
        vmin, vmax = None, None

    title = f"Magnetic Field Map ({result.model_name})"

    _fig, ax = plt.subplots(figsize=(8, 6))

    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, height * pixel_spacing_um, 0)

    im = ax.imshow(
        b_field,
        extent=extent,
        origin="upper",
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    _add_colorbar(im, ax, label=colorbar_label)

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
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
    logger.debug("Plotting parameter map: {} (model={})", param_name, result.model_name)
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
    extent = (0, width * pixel_spacing_um, height * pixel_spacing_um, 0)

    im = ax.imshow(
        param_map,
        extent=extent,
        origin="upper",
        cmap=cmap,
        aspect="equal",
    )

    _add_colorbar(im, ax, label=colorbar_label)

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

    The first panel shows B111 remanent (uT, diverging colormap) for multi-range
    models (ESR14N/ESR15N), or the legacy B-field (T) for single-range models.

    Args:
        result: FitResult or QDMResult containing fitted parameters.
        save: Whether to save the plot to file.
        filename: Custom filename for saving (optional).
    """
    logger.debug("Plotting fit result overview (model={})", result.model_name)
    plot_params = ["center", "width_0", "contrast", "chi2"]
    available_params = [p for p in plot_params if p in result.parameters]

    try:
        b_field = result.b111_remanent  # (H, W), uT
        b_title = "B111 remanent (µT)"
        b_cmap = "RdBu_r"
        b_vmax: float | None = float(np.nanpercentile(np.abs(b_field), 99))
        b_vmin: float | None = -b_vmax
    except (DataShapeError, AttributeError, TypeError):
        logger.warning("B111 remanent not available for overview, using legacy B-field calculation")
        b_field = result.calculate_b_field()  # (H, W), T
        b_title = "Magnetic Field (T)"
        b_cmap = "viridis"
        b_vmin, b_vmax = None, None

    n_plots = len(available_params) + 1  # +1 for B-field
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    _fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = [axes] if nrows == 1 and ncols == 1 else axes.flatten()

    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, height * pixel_spacing_um, 0)

    plot_idx = 0

    ax = axes[plot_idx]
    im = ax.imshow(
        b_field,
        extent=extent,
        origin="upper",
        cmap=b_cmap,
        aspect="equal",
        vmin=b_vmin,
        vmax=b_vmax,
    )
    ax.set_title(b_title)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    _add_colorbar(im, ax)
    plot_idx += 1

    for param in available_params:
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        param_map = result.get_parameter_map(param)

        im = ax.imshow(param_map, extent=extent, origin="upper", cmap="viridis", aspect="equal")
        ax.set_title(f"{param.replace('_', ' ').title()}")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        _add_colorbar(im, ax)
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


def plot_b111_map(
    result: FitResult,
    component: str = "remanent",
    *,
    save: bool = False,
    filename: str | None = None,
) -> None:
    """Plot one B111 component as a spatially-resolved map.

    A symmetric ``RdBu_r`` colormap is used and the colorbar limits are set
    to the 99th percentile of |B| so that a few outlier pixels do not
    dominate the scale.

    Args:
        result: FitResult or QDMResult with a ``b111`` property.
        component: Which component to plot: ``'remanent'`` or ``'induced'``.
        save: If True, save the figure to disk.
        filename: Output filename (auto-generated if None).

    Raises:
        ValueError: If component is not ``'remanent'`` or ``'induced'``.
    """
    logger.debug("Plotting B111 {} map", component)
    valid = {"remanent", "induced"}
    if component not in valid:
        msg = f"component must be one of {valid!r}, got {component!r}"
        raise ValueError(msg)

    b_map = result.b111[component].values  # (H, W), uT

    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, height * pixel_spacing_um, 0)

    vmax = float(np.nanpercentile(np.abs(b_map), 99))

    _fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        b_map,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-vmax,
        vmax=vmax,
    )
    _add_colorbar(im, ax, label=f"B111 {component} (µT)")
    ax.set_title(f"B111 {component} ({result.model_name})")
    _label_spatial_axes(ax)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"b111_{component}_{result.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
