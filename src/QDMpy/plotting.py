"""Visualization module for QDMpy.

This module provides a comprehensive set of plotting functions for visualizing data from
Quantum Diamond Microscopy (QDM) measurements. It includes capabilities for:

- Plotting raw and processed ODMR spectra
- Visualizing magnetic field maps
- Creating interactive figure handlers with browsable data
- Generating heatmaps and various other spatial visualizations
- Comparing original data with fitting results

The functions handle data in various formats, including multi-dimensional arrays
representing spatial, spectral, and polarization dimensions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from QDMpy.utils import double_norm

if TYPE_CHECKING:
    from QDMpy.result import FitResult


def plot_fit_result_field_map(
    result: FitResult, save: bool = False, filename: str | None = None
) -> None:
    """Plot magnetic field map from FitResult.

    Args:
        result: FitResult object containing fitted parameters
        save: Whether to save the plot to file
        filename: Custom filename for saving (optional)
        **kwargs: Additional arguments for plot customization
    """
    # Calculate magnetic field from fit results
    b_field = result.calculate_b_field()

    title = f"Magnetic Field Map ({result.model_name})"
    cmap = "viridis"
    colorbar_label = "Magnetic Field (T)"

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert pixel spacing to micrometers for axis labels
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

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    # Set labels and title
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
        **kwargs: Additional arguments for plot customization
    """
    # Get parameter data reshaped as 2D map
    param_map = result.get_parameter_map(param_name)

    # Set up default plot parameters based on parameter type
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

    title = f'{param_name.replace("_", " ").title()} Map ({result.model_name})'
    colorbar_label = param_labels.get(param_name, param_name.title())
    cmap = "viridis"

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert pixel spacing to micrometers for axis labels
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

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    # Set labels and title
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
        **kwargs: Additional arguments for plot customization
    """
    # Parameters to plot (if available)
    plot_params = ["center", "width_0", "contrast", "chi2"]
    available_params = [p for p in plot_params if p in result.parameters]

    # Add magnetic field
    b_field = result.calculate_b_field()

    n_plots = len(available_params) + 1  # +1 for B-field
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # Ensure axes is always a list for consistent indexing
    axes = [axes] if nrows == 1 and ncols == 1 else axes.flatten()

    # Convert pixel spacing to micrometers
    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, 0, height * pixel_spacing_um)

    plot_idx = 0

    # Plot magnetic field first
    ax = axes[plot_idx]
    im = ax.imshow(b_field, extent=extent, origin="lower", cmap="viridis", aspect="equal")
    ax.set_title("Magnetic Field (T)")
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    plt.colorbar(im, ax=ax)
    plot_idx += 1

    # Plot available parameters
    for param in available_params:
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        param_map = result.get_parameter_map(param)

        im = ax.imshow(param_map, extent=extent, origin="lower", cmap="viridis", aspect="equal")
        ax.set_title(f'{param.replace("_", " ").title()}')
        ax.set_xlabel("x [μm]")
        ax.set_ylabel("y [μm]")
        plt.colorbar(im, ax=ax)
        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Fit Results Overview ({result.model_name})", fontsize=14)
    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"fit_overview_{result.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_light_img(
    ax: plt.Axes,
    data: np.ndarray,
    img: matplotlib.image.AxesImage | None = None,
) -> matplotlib.image.AxesImage:
    """Plot light image on axes.

    Args:
        ax: Matplotlib axes to plot on.
        data: Image data to plot.
        img: Existing AxesImage to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Updated AxesImage object.
    """
    return update_img(
        ax,
        img,
        data,
        cmap="bone",
        interpolation="none",
        origin="lower",
        aspect="equal",
        zorder=0,
    )


def plot_fluorescence(
    ax: plt.Axes,
    data: np.ndarray,
    img: matplotlib.image.AxesImage | None = None,
) -> matplotlib.image.AxesImage:
    """Plot fluorescence image on axes.

    Args:
        ax: Matplotlib axes to plot on.
        data: Image data to plot.
        img: Existing AxesImage to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Updated AxesImage object.
    """
    return update_img(
        ax,
        img,
        data,
        cmap="inferno",
        interpolation="none",
        origin="lower",
        aspect="equal",
        zorder=0,
    )


def plot_laser_img(
    ax: plt.Axes,
    data: np.ndarray,
    img: matplotlib.image.AxesImage | None = None,
) -> matplotlib.image.AxesImage:
    """Plot laser image on axes.

    Args:
        ax: Matplotlib axes to plot on.
        data: Image data to plot.
        img: Existing AxesImage to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Updated AxesImage object.
    """
    return update_img(
        ax,
        img,
        data,
        cmap="magma",
        interpolation="none",
        origin="lower",
        aspect="equal",
        zorder=0,
    )


def update_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray | None = None,
    line: plt.Line2D | None = None,
    **plt_props: Any,  # noqa: ANN401
) -> plt.Line2D | None:
    """Update or create a line plot on axes.

    Args:
        ax: Matplotlib axes to plot on.
        x: X-axis data array.
        y: Y-axis data array (optional).
        line: Existing Line2D object to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Line2D object or None if y is None.
    """
    if y is None:
        return None
    if line is None:
        (line,) = ax.plot(x, y, **plt_props)
    elif all(y == line.get_ydata()):
        return line
    else:
        line.set_ydata(y)
    return line


def update_marker(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    line: plt.Line2D | None = None,
    **plt_props: Any,  # noqa: ANN401
) -> plt.Line2D:
    """Update or create a marker plot on axes.

    Args:
        ax: Matplotlib axes to plot on.
        x: X-axis data array.
        y: Y-axis data array.
        line: Existing Line2D object to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Updated or created Line2D object.
    """
    if line is None:
        (line,) = ax.plot(x, y, **plt_props)
    else:
        line.set_data(x, y)
    return line


def plot_quality_data(
    ax: plt.Axes,
    data: np.ndarray,
    img: matplotlib.image.AxesImage | None = None,
) -> matplotlib.image.AxesImage:
    """Plot quality data on axes with normalized colors.

    Args:
        ax: Matplotlib axes to plot on.
        data: Quality data to plot.
        img: Existing AxesImage to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Updated AxesImage object.
    """
    norm = get_color_norm(data.min(), data.max())
    return update_img(ax, img, data, norm=norm, cmap="inferno")


def plot_data(
    ax: plt.Axes,
    data: np.ndarray,
    img: matplotlib.image.AxesImage | None = None,
) -> matplotlib.image.AxesImage:
    """Plot data on axes with normalized colors.

    Args:
        ax: Matplotlib axes to plot on.
        data: Data to plot.
        img: Existing AxesImage to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Updated AxesImage object.
    """
    norm = get_color_norm(data.min(), data.max())
    return update_img(ax, img, data, norm=norm)


def get_vmin_vmax(
    img: matplotlib.image.AxesImage,
    percentile: float,
    use_percentile: bool,
) -> tuple[float, float]:
    """Get the vmin and vmax for the colorbar of the image.

    Args:
      img: matplotlib.image.AxesImage: The image to get the vmin and vmax from
      percentile: float: The percentile to use for the vmin and vmax
      use_percentile: bool: Whether to use the percentile or not

    Returns: Tuple[float, float]: The vmin and vmax

    """
    if img is None:
        return 0, 1

    data = img.get_array()
    if data is None:
        return 0, 1

    if percentile and use_percentile:
        vmin, vmax = np.percentile(
            data,
            [(100 - percentile) / 2, 100 - (100 - percentile) / 2],
        )
    else:
        vmin, vmax = (
            data.min(),
            data.max(),
        )
    return vmin, vmax


def get_color_norm(vmin: float, vmax: float) -> colors.Normalize:
    """Get appropriate color normalization for value range.

    Args:
        vmin: Minimum value.
        vmax: Maximum value.

    Returns:
        Matplotlib Normalize object (CenteredNorm or Normalize).
    """
    if vmin < 0 < vmax:
        return colors.CenteredNorm(halfrange=vmax, vcenter=0)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def plot_overlay(
    ax: plt.Axes,
    data: np.ndarray,
    img: matplotlib.image.AxesImage | None = None,
    normtype: str = "simple",
) -> matplotlib.image.AxesImage:
    """Plot overlay image with normalized alpha channel.

    Args:
        ax: Matplotlib axes to plot on.
        data: Overlay data to plot.
        img: Existing AxesImage to update (optional).
        normtype: Normalization type ("simple" by default).
        **plt_props: Additional plotting properties.

    Returns:
        Updated AxesImage object.
    """
    if normtype == "simple":
        alpha = double_norm(data)
    else:
        raise NotImplementedError(f"Normalization type {normtype} not implemented.")
    return update_img(ax, img, data, alpha=alpha)


def plot_outlier(
    ax: plt.Axes,
    data: np.ndarray,
    img: matplotlib.image.AxesImage | None = None,
) -> matplotlib.image.AxesImage:
    """Plot outlier mask on axes.

    Args:
        ax: Matplotlib axes to plot on.
        data: Outlier data to plot.
        img: Existing AxesImage to update (optional).
        **plt_props: Additional plotting properties.

    Returns:
        Updated AxesImage object.
    """
    data = data.astype(float)
    return update_img(ax, img, data, cmap="gist_rainbow", alpha=data, zorder=3)


def update_clim(
    img: matplotlib.image.AxesImage,
    vmin: float,
    vmax: float,
) -> matplotlib.image.AxesImage:
    """Update the colorbar limits of the image.

    Args:
      img: matplotlib.image.AxesImage: The image to update
      vmin: float: The new vmin
      vmax: float: The new vmax

    Returns: matplotlib.image.AxesImage: The updated image
    """
    norm = get_color_norm(vmin, vmax)
    img.set(norm=norm)
    return img


def update_cbar(
    img: matplotlib.image.AxesImage,
    cax: plt.Axes,
    vmin: float,
    vmax: float,
    original_cax_locator: Callable[..., Any],
) -> None:
    """Update colorbar limits and appearance.

    Args:
        img: AxesImage object to get data from.
        cax: Colorbar axes to update.
        vmin: Minimum colorbar value.
        vmax: Maximum colorbar value.
        original_cax_locator: Original axes locator to restore.
        **plt_props: Additional colorbar properties.
    """
    data = img.get_array()
    if data is None:
        mn, mx = 0, 1
    else:
        mn, mx = data.min(), data.max()

    extent = detect_extent(
        vmin=vmin,
        vmax=vmax,
        mn=mn,
        mx=mx,
    )

    label = cax.get_ylabel()
    cax.clear()
    cax.set_axes_locator(original_cax_locator)
    plt.colorbar(img, cax=cax, extend=extent, label=label)


def detect_extent(vmin: float, vmax: float, mn: float, mx: float) -> str:
    """Detects the extend of the colorbar.

    Args:
      vmin: float: minimum value of the colorbar
      vmax: float: maximum value of the colorbar
      mn: float: minimum value of the data
      mx: float: maximum value of the data

    Returns: str: "neither", "min", "max", "both"
    """
    if vmin == mn and vmax == mx:
        return "neither"
    if vmin > mn and vmax < mx:
        return "both"
    if vmin > mn:
        return "min"
    return "max"


def update_img(
    ax: plt.Axes,
    img: matplotlib.image.AxesImage | None,
    data: np.ndarray,
    **plt_props: Any,  # noqa: ANN401
) -> matplotlib.image.AxesImage:
    """Update or create image plot on axes.

    Args:
        ax: Matplotlib axes to plot on.
        img: Existing AxesImage to update (optional).
        data: Image data to plot.
        **plt_props: Additional plotting properties.

    Returns:
        Updated or created AxesImage object.
    """
    data_dimensions = plt_props.pop("data_dimensions", data.shape)
    plt_props["extent"] = [0, data_dimensions[1], 0, data_dimensions[0]]
    plt_props["origin"] = "lower"
    plt_props["aspect"] = "equal"
    if img is None:
        img = ax.imshow(data, **plt_props)
    else:
        if "alpha" in plt_props:
            img.set_alpha(plt_props["alpha"])
        img.set_data(data)
    return img


def toggle_img(img: matplotlib.image.AxesImage | None = None) -> None:
    """Toggle visibility of image.

    Args:
        img: AxesImage object to toggle (optional).
    """
    if img is None:
        return
    img.set_visible(not img.get_visible())


