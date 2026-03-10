"""Composite QDM display: measurement images, pixel spectra, and overview."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.exceptions import DataShapeError, ParameterError
from qdmpy.plotting._common import (
    _add_colorbar,
    _avg_param_map,
    _finalize_layout,
    _label_spatial_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes

    from qdmpy.fitting.models import Model
    from qdmpy.fitting.result import FitResult
    from qdmpy.measurement import Measurement
    from qdmpy.result import QDMResult


def plot_measurement_images(measurement: Measurement) -> None:
    """Plot the light and laser optical images from a Measurement.

    Args:
        measurement: Measurement instance containing ``light_image`` and
            ``laser_image`` arrays.
    """
    logger.debug("Plotting measurement optical images")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(measurement.light_image, cmap="gray", origin="upper", aspect="equal")
    axes[0].set_title("Light image")
    axes[0].axis("off")

    axes[1].imshow(measurement.laser_image, cmap="gray", origin="upper", aspect="equal")
    axes[1].set_title("Laser image")
    axes[1].axis("off")

    fig.suptitle("Optical images", fontsize=12)
    _finalize_layout(fig, reserve_top=0.05)
    plt.show()


def _draw_fit_curve(
    ax: MplAxes,
    fit_result: FitResult,
    freq: NDArray,
    i_pol: int,
    i_frange: int,
    flat_idx: int,
    model: Model,
    color: str,
    ls: str,
) -> None:
    """Overlay one (pol, frange) model fit curve onto *ax*, silently skipping on error.

    Args:
        ax: Target axes.
        fit_result: FitResult providing parameter arrays.
        freq: 1-D frequency array in GHz for this freq_range.
        i_pol: Polarity index.
        i_frange: Frequency-range index.
        flat_idx: Flat pixel index into the n_pixel dimension.
        model: Model instance used to evaluate the fit curve.
        color: Line colour.
        ls: Line style.
    """
    try:
        params_arr = np.array(
            [
                fit_result.parameters[pn][i_pol, i_frange, flat_idx]
                for pn in model.parameter_names
                if pn in fit_result.parameters
            ]
        ).reshape(1, -1)
        fit_curve = model.func(freq, params_arr)[0]
        ax.plot(freq, fit_curve, color=color, ls=ls, lw=2.0)
    except (IndexError, ValueError, KeyError) as exc:
        logger.debug("Fit curve evaluation failed for pixel {}: {}", flat_idx, exc)


def _draw_pixel_spectra(
    ax: MplAxes,
    fit_result: FitResult,
    freq_ghz: NDArray,
    data_values: NDArray,
    y_idx: int,
    x_idx: int,
    flat_idx: int,
    model: Model | None,
) -> None:
    """Draw all (pol, frange) raw spectra and optional fit curves onto *ax*.

    Args:
        ax: Target axes panel.
        fit_result: FitResult providing parameter arrays.
        freq_ghz: Frequency array with shape (n_frange, n_freq) in GHz.
        data_values: ODMR data with shape (n_pol, n_frange, y, x, n_freq).
        y_idx: Row index for the selected pixel.
        x_idx: Column index for the selected pixel.
        flat_idx: Flat pixel index into the n_pixel dimension.
        model: Model instance (or None) for fit-curve evaluation.
    """
    n_pol = data_values.shape[0]
    n_frange = data_values.shape[1]
    pol_colors = ["tab:blue", "tab:red"]
    frange_ls = ["-", "--"]

    for i_pol in range(n_pol):
        for i_frange in range(n_frange):
            freq = freq_ghz[i_frange]
            spectrum = data_values[i_pol, i_frange, y_idx, x_idx, :]
            color = pol_colors[i_pol % len(pol_colors)]
            ls = frange_ls[i_frange % len(frange_ls)]
            ax.plot(
                freq,
                spectrum,
                color=color,
                ls=ls,
                lw=0.8,
                alpha=0.5,
                label=f"pol={i_pol} fr={i_frange}",
            )
            if model is not None:
                _draw_fit_curve(ax, fit_result, freq, i_pol, i_frange, flat_idx, model, color, ls)

    ax.set_title(f"Pixel ({y_idx}, {x_idx})")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=7, loc="lower right")


def _plot_display_pixel_spectra(
    axes: NDArray,
    start_row: int,
    n_sample_pixels: int,
    fit_result: FitResult,
    measurement: Measurement,
    n_cols: int,
) -> None:
    """Fill axes rows with representative pixel ODMR spectra plus fit curves.

    Pixels are chosen at equally-spaced percentile positions of the flattened
    B111 remanent map so the sample covers the full dynamic range of the field.

    Args:
        axes: 2-D axes array from the parent ``plt.subplots`` call.
        start_row: Row index where spectra panels begin.
        n_sample_pixels: Total number of pixel panels to fill.
        fit_result: FitResult with fitted parameters.
        measurement: Measurement containing processed ODMR data.
        n_cols: Number of columns in the axes grid.
    """
    from qdmpy.exceptions import DataNotLoadedError

    try:
        odmr_data = measurement.odmr.processed_data
    except (AttributeError, DataNotLoadedError):
        logger.warning("Processed ODMR data not available, skipping pixel spectra overlay")
        return

    _, width = fit_result.scan_dimensions

    b111_flat = fit_result.b111_remanent.flatten()
    percs = np.linspace(10, 90, n_sample_pixels)
    flat_indices = [int(np.argmin(np.abs(b111_flat - np.percentile(b111_flat, p)))) for p in percs]

    model: Model | None = None
    try:
        from qdmpy.fitting.models import ModelRegistry

        model = ModelRegistry.get(fit_result.model_name)
    except KeyError as exc:
        logger.debug("Model {} not available for fit-curve overlay: {}", fit_result.model_name, exc)

    freq_ghz = odmr_data.data.coords["freq_ghz"].values  # (n_frange, n_freq)
    data_values = odmr_data.data.values  # (n_pol, n_frange, y, x, n_freq)

    for i_pixel, flat_idx in enumerate(flat_indices):
        row = start_row + i_pixel // n_cols
        col = i_pixel % n_cols
        if row >= axes.shape[0] or col >= axes.shape[1]:
            break
        y_idx, x_idx = divmod(flat_idx, width)
        _draw_pixel_spectra(
            axes[row, col],
            fit_result,
            freq_ghz,
            data_values,
            y_idx,
            x_idx,
            flat_idx,
            model,
        )

    for i_pixel in range(len(flat_indices), n_sample_pixels):
        row = start_row + i_pixel // n_cols
        col = i_pixel % n_cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)


def _compute_display_layout(
    height: int,
    width: int,
    has_images: bool,
    spec_rows: int,
) -> tuple[tuple[float, float], list[float]]:
    """Compute figure size and row-height ratios for plot_qdm_display.

    The map rows use equal-aspect images, so their preferred height depends on
    scan aspect ratio. Spectra rows are intentionally shorter to avoid very tall,
    sparse line plots.
    """
    n_cols = 3
    image_aspect = width / max(height, 1)

    col_width = 3.6
    map_row_height = float(np.clip(col_width / max(image_aspect, 0.25), 2.2, 4.8))
    optical_row_height = map_row_height * 0.9
    spectra_row_height = 2.2

    row_heights: list[float] = [map_row_height, map_row_height]
    if has_images:
        row_heights.append(optical_row_height)
    if spec_rows > 0:
        row_heights.extend([spectra_row_height] * spec_rows)

    fig_width = n_cols * col_width + 0.4
    fig_height = sum(row_heights) + 0.8
    return (fig_width, fig_height), row_heights


def _draw_b111_row(
    axes: NDArray,
    fit_result: FitResult,
    extent: tuple[float, float, float, float],
    height: int,
    width: int,
) -> None:
    """Draw B111 remanent, B111 induced, and chi-squared maps into row 0.

    Falls back to a mean centre-frequency map when B111 cannot be computed
    (e.g. single-polarity data).

    Args:
        axes: 2-D axes grid.
        fit_result: FitResult providing b111 and chi2.
        extent: imshow extent tuple (left, right, bottom, top) in um with origin='upper'.
        height: Scan height in pixels.
        width: Scan width in pixels.
    """
    try:
        b_rem = fit_result.b111_remanent  # (height, width), uT
        b_ind = fit_result.b111_induced
        vmax_rem = float(np.nanpercentile(np.abs(b_rem), 99))
        im = axes[0, 0].imshow(
            b_rem,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-vmax_rem,
            vmax=vmax_rem,
        )
        axes[0, 0].set_title("B111 remanent (µT)")
        _label_spatial_axes(axes[0, 0])
        _add_colorbar(im, axes[0, 0])
        vmax_ind = float(np.nanpercentile(np.abs(b_ind), 99))
        im = axes[0, 1].imshow(
            b_ind,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-vmax_ind,
            vmax=vmax_ind,
        )
        axes[0, 1].set_title("B111 induced (µT)")
        _label_spatial_axes(axes[0, 1])
        _add_colorbar(im, axes[0, 1])
    except (DataShapeError, AttributeError, TypeError):
        logger.warning("B111 not available (single polarity?), showing mean centre map instead")
        # Single-polarity data: show mean centre map as fallback
        center_fb = _avg_param_map(fit_result.centers, height, width)
        im = axes[0, 0].imshow(
            center_fb, extent=extent, origin="upper", cmap="viridis", aspect="equal"
        )
        axes[0, 0].set_title("Centre (GHz)")
        _label_spatial_axes(axes[0, 0])
        _add_colorbar(im, axes[0, 0])
        axes[0, 1].set_visible(False)

    chi2_map = _avg_param_map(fit_result.chi2, height, width)
    im = axes[0, 2].imshow(chi2_map, extent=extent, origin="upper", cmap="magma", aspect="equal")
    axes[0, 2].set_title("Chi-squared")
    _label_spatial_axes(axes[0, 2])
    _add_colorbar(im, axes[0, 2])


def _draw_param_row(
    axes: NDArray,
    fit_result: FitResult,
    extent: tuple[float, float, float, float],
    height: int,
    width: int,
) -> None:
    """Draw mean centre, contrast, and linewidth maps into row 1.

    Args:
        axes: 2-D axes grid.
        fit_result: FitResult providing centres, contrasts, and linewidths.
        extent: imshow extent tuple in um.
        height: Scan height in pixels.
        width: Scan width in pixels.
    """
    center_map = _avg_param_map(fit_result.centers, height, width)
    im = axes[1, 0].imshow(
        center_map, extent=extent, origin="upper", cmap="viridis", aspect="equal"
    )
    axes[1, 0].set_title("Centre (GHz, mean)")
    _label_spatial_axes(axes[1, 0])
    _add_colorbar(im, axes[1, 0])

    try:
        contrast_map = _avg_param_map(fit_result.contrasts, height, width)
    except ParameterError:
        logger.warning("Contrast parameter not available, using zeros")
        contrast_map = np.zeros((height, width))
    im = axes[1, 1].imshow(
        contrast_map, extent=extent, origin="upper", cmap="viridis", aspect="equal"
    )
    axes[1, 1].set_title("Contrast (mean)")
    _label_spatial_axes(axes[1, 1])
    _add_colorbar(im, axes[1, 1])

    try:
        lw_map = _avg_param_map(fit_result.linewidths, height, width)
    except ParameterError:
        logger.warning("Linewidth parameter not available, using zeros")
        lw_map = np.zeros((height, width))
    im = axes[1, 2].imshow(lw_map, extent=extent, origin="upper", cmap="viridis", aspect="equal")
    axes[1, 2].set_title("Linewidth (GHz, mean)")
    _label_spatial_axes(axes[1, 2])
    _add_colorbar(im, axes[1, 2])


def plot_qdm_display(
    result: FitResult | QDMResult,
    measurement: Measurement | None = None,
    n_sample_pixels: int = 3,
) -> None:
    """Comprehensive overview display for a QDM fit result.

    Always shown:
      - B111 remanent and induced maps (uT, diverging colormap)
      - Chi-squared map
      - Mean resonance centre, contrast, and linewidth maps

    Shown when optical images are available (from result or measurement):
      - Light and laser optical images
      - ``n_sample_pixels`` representative pixel spectra with fit curves
        (only when measurement is provided for raw ODMR data access)

    Optical images are sourced in priority order:
      1. ``result.light_image`` / ``result.laser_image`` (QDMResult fields)
      2. ``measurement.light_image`` / ``measurement.laser_image`` (fallback)

    Pixels for spectral display are chosen at equally-spaced percentile
    positions of B111 remanent so they sample the full dynamic range.

    Args:
        result: FitResult or QDMResult.
        measurement: Optional Measurement for ODMR spectra and image fallback.
        n_sample_pixels: Number of sample pixel spectra (default 3).
    """
    from qdmpy.result import QDMResult as _QDMResult

    fit_result = result.fit_result if isinstance(result, _QDMResult) else result
    logger.info("Plotting QDM display overview (model={})", fit_result.model_name)

    # Resolve optical images: QDMResult fields take priority over measurement
    light_image = None
    laser_image = None
    if isinstance(result, _QDMResult):
        light_image = result.light_image
        laser_image = result.laser_image
    if light_image is None and measurement is not None:
        light_image = measurement.light_image
    if laser_image is None and measurement is not None:
        laser_image = measurement.laser_image

    has_images = light_image is not None or laser_image is not None

    height, width = fit_result.scan_dimensions
    pixel_spacing_um = fit_result.pixel_spacing * 1e6
    extent = (0, width * pixel_spacing_um, height * pixel_spacing_um, 0)

    spec_rows = -(-n_sample_pixels // 3) if measurement is not None else 0  # ceil div
    n_rows = 2 + (1 if has_images else 0) + spec_rows
    n_cols = 3

    figsize, height_ratios = _compute_display_layout(height, width, has_images, spec_rows)

    fig, axes_raw = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes: NDArray = np.atleast_2d(axes_raw)

    _draw_b111_row(axes, fit_result, extent, height, width)
    _draw_param_row(axes, fit_result, extent, height, width)

    if has_images:
        row = 2
        if light_image is not None:
            axes[row, 0].imshow(light_image, cmap="gray", origin="upper", aspect="equal")
            axes[row, 0].set_title("Light image")
        else:
            axes[row, 0].set_visible(False)
        axes[row, 0].axis("off")
        if laser_image is not None:
            axes[row, 1].imshow(laser_image, cmap="gray", origin="upper", aspect="equal")
            axes[row, 1].set_title("Laser image")
        else:
            axes[row, 1].set_visible(False)
        axes[row, 1].axis("off")
        axes[row, 2].set_visible(False)
        if measurement is not None:
            _plot_display_pixel_spectra(
                axes, row + 1, n_sample_pixels, fit_result, measurement, n_cols
            )

    fig.suptitle(f"QDM Result Overview ({fit_result.model_name})", fontsize=14)
    _finalize_layout(fig, reserve_top=0.06)
    plt.show()
