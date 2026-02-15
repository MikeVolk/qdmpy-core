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

import itertools
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from QDMpy.utils import double_norm

if TYPE_CHECKING:
    from QDMpy.measurement import Measurement
    from QDMpy.result import FitResult

# Import for runtime usage
from QDMpy import models

FREQ_LABEL = "f [GHz]"
CONTRAST_LABEL = "c [%]"


def plot_fit_result_field_map(
    result: FitResult, save: bool = False, filename: str | None = None, **kwargs: Any
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

    # Set up default plot parameters
    plot_kwargs = {
        "title": f"Magnetic Field Map ({result.model_name})",
        "pixel_spacing": result.pixel_spacing,
        "colorbar_label": "Magnetic Field (T)",
        "cmap": "viridis",
        **kwargs,
    }

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
        cmap=plot_kwargs.get("cmap", "viridis"),
        aspect="equal",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(plot_kwargs.get("colorbar_label", "Magnetic Field (T)"))

    # Set labels and title
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title(plot_kwargs.get("title", "Magnetic Field Map"))

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
    **kwargs: Any,
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

    default_title = f'{param_name.replace("_", " ").title()} Map ({result.model_name})'
    default_colorbar_label = param_labels.get(param_name, param_name.title())

    plot_kwargs = {
        "title": default_title,
        "colorbar_label": default_colorbar_label,
        "cmap": "viridis",
        **kwargs,
    }

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
        cmap=plot_kwargs.get("cmap", "viridis"),
        aspect="equal",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(plot_kwargs.get("colorbar_label", param_name.title()))

    # Set labels and title
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title(plot_kwargs.get("title", f"{param_name} Map"))

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"{param_name}_map_{result.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_fit_result_overview(
    result: FitResult, save: bool = False, filename: str | None = None, **kwargs: Any
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
    img: mpl.image.AxesImage | None = None,
    **plt_props: Any | None,
) -> mpl.image.AxesImage:
    """Args:
      ax:
      data:
      img:  (Default value = None)
      **plt_props:

    Returns:

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
        **plt_props,
    )


def plot_fluorescence(
    ax: plt.Axes,
    data: np.ndarray,
    img: mpl.image.AxesImage | None = None,
    **plt_props: Any | None,
) -> mpl.image.AxesImage:
    """Args:
      ax:
      data:
      img:  (Default value = None)
      **plt_props:

    Returns:

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
        **plt_props,
    )


def plot_laser_img(
    ax: plt.Axes,
    data: np.ndarray,
    img: mpl.image.AxesImage | None = None,
    **plt_props: Any,
) -> mpl.image.AxesImage:
    """Args:
      ax: plt.Axes:
      data:
      img:  (Default value = None)
      **plt_props:

    Returns:

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
        **plt_props,
    )


def update_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray | None = None,
    line: plt.Line2D | None = None,
    **plt_props: Any,
) -> plt.Line2D | None:
    """Args:
      ax: plt.Axes:
      x:np.ndarray[float]:
      y:np.ndarray[float]:  (Default value = None)
      line:plt.Line2D:  (Default value = None)
      **plt_props:

    Returns:

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
    **plt_props: Any,
) -> plt.Line2D:
    """Args:
      ax: plt.Axes:
      x:
      y:
      line:  (Default value = None)
      **plt_props:

    Returns:

    """
    if line is None:
        (line,) = ax.plot(x, y, **plt_props)
    else:
        line.set_data(x, y)
    return line


def plot_quality_data(
    ax: plt.Axes,
    data: np.ndarray,
    img: mpl.image.AxesImage | None = None,
    **plt_props: Any,
) -> mpl.image.AxesImage:
    """Args:
      ax: plt.Axes:
      data:
      img:  (Default value = None)
      **plt_props:

    Returns:

    """
    norm = get_color_norm(data.min(), data.max())
    plt_props["norm"] = norm
    plt_props["cmap"] = "inferno"
    return update_img(ax, img, data, **plt_props)


def plot_data(
    ax: plt.Axes,
    data: np.ndarray,
    img: mpl.image.AxesImage | None = None,
    **plt_props: Any,
) -> mpl.image.AxesImage:
    """Args:
      ax: plt.Axes:
      data:
      img:  (Default value = None)
      **plt_props:

    Returns:

    """
    norm = get_color_norm(data.min(), data.max())
    plt_props["norm"] = norm
    return update_img(ax, img, data, **plt_props)


def get_vmin_vmax(
    img: mpl.image.AxesImage,
    percentile: float,
    use_percentile: bool,
) -> tuple[float, float]:
    """Get the vmin and vmax for the colorbar of the image.

    Args:
      img: mpl.image.AxesImage: The image to get the vmin and vmax from
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
    """Args:
      vmin:
      vmax:

    Returns:

    """
    if vmin < 0 < vmax:
        return colors.CenteredNorm(halfrange=vmax, vcenter=0)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def plot_overlay(
    ax: plt.Axes,
    data: np.ndarray,
    img: mpl.image.AxesImage | None | None = None,
    normtype: str = "simple",
    **plt_props: Any,
) -> mpl.image.AxesImage:
    """Args:
      ax: plt.Axes:
      data:
      img:  (Default value = None)
      normtype:  (Default value = "simple")
      **plt_props:

    Returns:

    """
    if normtype == "simple":
        plt_props["alpha"] = double_norm(data)
    else:
        raise NotImplementedError(f"Normalization type {normtype} not implemented.")
    return update_img(ax, img, data, **plt_props)


def plot_outlier(
    ax: plt.Axes,
    data: np.ndarray,
    img: mpl.image.AxesImage | None = None,
    **plt_props: Any,
) -> mpl.image.AxesImage:
    """Args:
      ax: plt.Axes:
      data:
      img:  (Default value = None)
      **plt_props:

    Returns:

    """
    data = data.astype(float)
    plt_props["cmap"] = "gist_rainbow"
    plt_props["alpha"] = data
    plt_props["zorder"] = 3
    return update_img(ax, img, data, **plt_props)


def update_clim(
    img: mpl.image.AxesImage,
    vmin: float,
    vmax: float,
) -> mpl.image.AxesImage:
    """Update the colorbar limits of the image.

    Args:
      img: mpl.image.AxesImage: The image to update
      vmin: float: The new vmin
      vmax: float: The new vmax

    Returns: mpl.image.AxesImage: The updated image
    """
    norm = get_color_norm(vmin, vmax)
    img.set(norm=norm)
    return img


def update_cbar(
    img: mpl.image.AxesImage,
    cax: plt.Axes,
    vmin: float,
    vmax: float,
    original_cax_locator: Any,
    **plt_props: dict,
) -> None:
    """Args:
      img:
      cax:
      vmin:
      vmax:
      original_cax_locator:

    Returns:

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
    plt.colorbar(img, cax=cax, extend=extent, label=label, **plt_props)


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
    img: mpl.image.AxesImage | None,
    data: np.ndarray,
    **plt_props: Any,
) -> mpl.image.AxesImage:
    """Args:
      ax: plt.Axes:
      img:
      data:
      **plt_props:

    Returns:

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


def toggle_img(img: mpl.image.AxesImage | None = None) -> None:
    """Args:
      img:  (Default value = None).

    Returns:

    """
    if img is None:
        return
    img.set_visible(not img.get_visible())


def check_fit_pixel(qdm_obj: Measurement, idx: int) -> tuple[plt.Figure, plt.Axes]:
    """Args:
      qdm_obj:
      idx:

    Returns:

    """
    # noinspection PyTypeChecker
    f, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=True)
    polarities = ["+", "-"]
    model = [None, models.esrsingle, models.esr15n, models.esr14n][qdm_obj.model_name]

    for p, frange in itertools.product(
        range(qdm_obj.odmr.n_pol),
        range(qdm_obj.odmr.n_frange),
    ):
        f_new = np.linspace(min(qdm_obj.odmr.f_ghz[frange]), max(qdm_obj.odmr.f_ghz[frange]), 200)

        m_initial = model(parameter=qdm_obj.fit.initial_parameter[p, frange, [idx]], x=f_new)
        m_fit = model(parameter=qdm_obj.fit.model_params[p, frange, [idx]], x=f_new)

        ax[frange].plot(
            qdm_obj.odmr.f_ghz[frange],
            qdm_obj.odmr.data[p, frange, [idx]][0],
            "k",
            marker=["o", "^"][p],
            markersize=5,
            mfc="w",
            label=f"data: {polarities[p]}",
            ls="",
        )
        (line,) = ax[frange].plot(f_new, m_initial[0], label="initial guess", alpha=0.5, ls=":")
        ax[frange].plot(f_new, m_fit[0], color=line.get_color(), label="fit")
        ax[frange].legend(
            ncol=2,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            mode="expand",
            borderaxespad=0.0,
        )

        line = " ".join([f"{v:>8.5f}" for v in qdm_obj.fit.model_params[p, frange, idx]])
        line += f" {qdm_obj.fit._chi_squares[p, frange, idx]:>8.2e}"

    for a in ax.flat:
        a.set(xlabel=FREQ_LABEL, ylabel="ODMR contrast [a.u.]")
    return f, ax


def plot_fit_params(
    qdm_obj: Measurement,
    param: str,
    save: str | bool = False,
) -> plt.Figure:
    """Args:
      qdm_obj:
      param:
      save:  (Default value = False).

    Returns:

    """
    data = qdm_obj.get_param(param)

    if param == "contrast":
        data = data.mean(axis=2)
    if "contrast" in param:
        data *= 100
    if param == "width":
        data *= 1000

    labels = {
        "center": FREQ_LABEL,
        "resonance": FREQ_LABEL,
        "width": "f [MHz]",
        "contrast": "mean(c) [%]",
        "contrast_0": CONTRAST_LABEL,
        "contrast_1": CONTRAST_LABEL,
        "contrast_2": CONTRAST_LABEL,
        "chi2": "chi$^2$",
    }

    # noinspection PyTypeChecker
    f, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    f.suptitle(f"{param}")

    # determine min and max of the plot
    vminl = np.min(np.sort(data[:, 0].flat)[50:-50])
    vmaxl = np.max(np.sort(data[:, 0].flat)[50:-50])
    vminr = np.min(np.sort(data[:, 1].flat)[50:-50])
    vmaxr = np.max(np.sort(data[:, 1].flat)[50:-50])

    # positive field direction
    ax[0, 0].set_title(r"B$^+_\mathrm{lf}$")
    ax[0, 0].imshow(data[0, 0], origin="lower", vmin=vminl, vmax=vmaxl)
    ax[0, 1].set_title(r"B$^+_\mathrm{hf}$")
    ax[0, 1].imshow(data[0, 1], origin="lower", vmin=vminr, vmax=vmaxr)

    # negative field direction
    ax[1, 0].set_title(r"B$^-_\mathrm{lf}$")
    c = ax[1, 0].imshow(data[1, 0], origin="lower", vmin=vminl, vmax=vmaxl)
    cb = plt.colorbar(c, ax=ax[:, 0], shrink=0.9)
    cb.ax.set_ylabel(labels[param])

    ax[1, 1].set_title(r"B$^-_\mathrm{hf}$")
    c = ax[1, 1].imshow(data[1, 1], origin="lower", vmin=vminr, vmax=vmaxr)
    cb = plt.colorbar(c, ax=ax[:, 1], shrink=0.9)
    cb.ax.set_ylabel(labels[param])

    for a in ax.flat:
        a.set(xlabel="px", ylabel="px")

    if save:
        f.savefig(save)
    return f
