"""Visualization module for QDMpy.

This module provides plotting functions for visualizing data from Quantum Diamond
Microscopy (QDM) measurements, including magnetic field maps and spatial parameter maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.constants import D_ZFS
from qdmpy.utils import double_norm

if TYPE_CHECKING:
    import matplotlib.figure
    from matplotlib.axes import Axes as MplAxes

    from qdmpy.fitting.models import Model
    from qdmpy.fitting.result import FitResult
    from qdmpy.magnetic_map import MagneticMap
    from qdmpy.measurement import Measurement
    from qdmpy.odmr.data import ODMRData
    from qdmpy.odmr.folding import FoldedODMR
    from qdmpy.result import QDMResult

# Set white background for all QDMpy figures
mpl.rcParams["figure.facecolor"] = "white"


def plot_fit_result_field_map(
    result: FitResult, save: bool = False, filename: str | None = None
) -> None:
    """Plot magnetic field map from FitResult.

    Uses B111 remanent (µT, diverging colormap) for multi-range models such as
    ESR14N/ESR15N.  Falls back to the legacy ``calculate_b_field()`` (T) for
    single-range models where B111 decomposition is not available.

    Args:
        result: FitResult or QDMResult containing fitted parameters.
        save: Whether to save the plot to file.
        filename: Custom filename for saving (optional).
    """
    try:
        b_field = result.b111_remanent  # (H, W), µT
        colorbar_label = "B111 remanent (µT)"
        cmap = "RdBu_r"
        vmax: float | None = float(np.nanpercentile(np.abs(b_field), 99))
        vmin: float | None = -vmax
    except Exception:
        b_field = result.calculate_b_field()  # (H, W), T
        colorbar_label = "Magnetic Field (T)"
        cmap = "viridis"
        vmin, vmax = None, None

    title = f"Magnetic Field Map ({result.model_name})"

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
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

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

    The first panel shows B111 remanent (µT, diverging colormap) for multi-range
    models (ESR14N/ESR15N), or the legacy B-field (T) for single-range models.

    Args:
        result: FitResult or QDMResult containing fitted parameters.
        save: Whether to save the plot to file.
        filename: Custom filename for saving (optional).
    """
    plot_params = ["center", "width_0", "contrast", "chi2"]
    available_params = [p for p in plot_params if p in result.parameters]

    try:
        b_field = result.b111_remanent  # (H, W), µT
        b_title = "B111 remanent (µT)"
        b_cmap = "RdBu_r"
        b_vmax: float | None = float(np.nanpercentile(np.abs(b_field), 99))
        b_vmin: float | None = -b_vmax
    except Exception:
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
    extent = (0, width * pixel_spacing_um, 0, height * pixel_spacing_um)

    plot_idx = 0

    ax = axes[plot_idx]
    im = ax.imshow(
        b_field,
        extent=extent,
        origin="lower",
        cmap=b_cmap,
        aspect="equal",
        vmin=b_vmin,
        vmax=b_vmax,
    )
    ax.set_title(b_title)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    plt.colorbar(im, ax=ax)
    plot_idx += 1

    for param in available_params:
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        param_map = result.get_parameter_map(param)

        im = ax.imshow(param_map, extent=extent, origin="lower", cmap="viridis", aspect="equal")
        ax.set_title(f"{param.replace('_', ' ').title()}")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
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


# ---------------------------------------------------------------------------
# Folding diagnostic plots
# ---------------------------------------------------------------------------


def plot_folding_search_landscape(folded: FoldedODMR) -> None:
    """Plot the D_ZFS brute-force search residual vs candidate.

    One subplot per polarity. The minimum is marked with a vertical dashed line.

    Args:
        folded: FoldedODMR result with d_candidates and search_residual populated.
    """
    if folded.d_candidates is None or folded.search_residual is None:
        return

    d_cand = folded.d_candidates
    residual = folded.search_residual  # (n_pol, n_steps)
    n_pol = residual.shape[0]
    pol_labels = list(folded.d_zfs_map.coords["polarity"].values)

    _fig, axes = plt.subplots(1, n_pol, figsize=(5 * n_pol, 4), squeeze=False)

    for i_pol in range(n_pol):
        ax = axes[0, i_pol]
        res = residual[i_pol]
        valid = np.isfinite(res)
        ax.plot(d_cand[valid], res[valid], "k-", lw=1)

        best_idx = np.argmin(res)
        ax.axvline(
            d_cand[best_idx],
            color="tab:red",
            ls="--",
            lw=1.5,
            label=f"min @ {d_cand[best_idx]:.6f} GHz",
        )
        ax.set_xlabel("D candidate (GHz)")
        ax.set_ylabel("Mean fold residual")
        ax.set_title(f"D_ZFS search -- pol={pol_labels[i_pol]}")
        ax.legend(fontsize=8)

    plt.suptitle("D_ZFS brute-force search", fontsize=12)
    plt.tight_layout()
    plt.show()


def resolve_pixel_indices(
    n_y: int,
    n_x: int,
    x: list[int] | int | None = None,
    y: list[int] | int | None = None,
) -> list[tuple[int, int]]:
    """Resolve pixel coordinate arguments into a list of (y, x) index pairs.

    Expansion rules:
    - Both None: one random pixel is chosen.
    - Both scalar: single pixel ``[(y, x)]``.
    - x is list, y is scalar: ``[(y, x0), (y, x1), ...]``.
    - x is scalar, y is list: ``[(y0, x), (y1, x), ...]``.
    - Both lists (same length): ``zip(y, x)``.

    Args:
        n_y: Height of the scan grid (number of rows).
        n_x: Width of the scan grid (number of columns).
        x: Column index or indices. None means random.
        y: Row index or indices. None means random.

    Returns:
        List of (row, col) index pairs.

    Raises:
        ValueError: If both x and y are lists with mismatched lengths.
    """
    if x is None and y is None:
        rand_y = int(np.random.randint(0, n_y))
        rand_x = int(np.random.randint(0, n_x))
        return [(rand_y, rand_x)]

    x_list: list[int] = [x] if isinstance(x, int) else (list(x) if x is not None else [])
    y_list: list[int] = [y] if isinstance(y, int) else (list(y) if y is not None else [])

    # Scalar x with list y, or vice versa
    if isinstance(y, list) and isinstance(x, int):
        return [(yi, x) for yi in y]
    if isinstance(x, list) and isinstance(y, int):
        return [(y, xi) for xi in x]
    if isinstance(x, list) and isinstance(y, list):
        if len(x_list) != len(y_list):
            msg = f"x and y lists must have the same length, got {len(x_list)} vs {len(y_list)}"
            raise ValueError(msg)
        return list(zip(y_list, x_list, strict=True))

    # Both scalars or one-element case
    yi = y_list[0] if y_list else int(np.random.randint(0, n_y))
    xi = x_list[0] if x_list else int(np.random.randint(0, n_x))
    return [(yi, xi)]


def plot_folding_pixel_spectra(
    folded: FoldedODMR,
    x: list[int] | int | None = None,
    y: list[int] | int | None = None,
) -> None:
    """Plot folded, unfolded, and antisymmetric spectra for one or more pixels.

    Works like ``plot_folding_mean_spectrum`` but shows individual pixel traces
    instead of the spatial mean. One subplot per polarity; each selected pixel
    gets its own colour. A random pixel is used when neither x nor y is given.

    Expansion rules for x / y arguments:
    - Both None -> one random pixel.
    - Both scalar -> single pixel.
    - x list + y scalar -> ``[(y, x0), (y, x1), ...]``.
    - x scalar + y list -> ``[(y0, x), (y1, x), ...]``.
    - Both lists (same length) -> ``zip(y, x)``.

    Args:
        folded: FoldedODMR result.
        x: Column index or list of column indices. None for random.
        y: Row index or list of row indices. None for random.
    """
    n_y = folded.folded_spectrum.sizes["y"]
    n_x = folded.folded_spectrum.sizes["x"]
    pixels = resolve_pixel_indices(n_y, n_x, x=x, y=y)

    delta_f = folded.folded_spectrum.coords["delta_f_ghz"].values  # (n_df,)
    delta_f_mhz = delta_f * 1000

    pol_labels = list(folded.folded_spectrum.coords["polarity"].values)
    n_pol = len(pol_labels)

    _fig, axes = plt.subplots(1, n_pol, figsize=(6 * n_pol, 4), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i_pol in range(n_pol):
        ax = axes[0, i_pol]
        ax2 = ax.twinx()

        for idx, (yi, xi) in enumerate(pixels):
            color = colors[idx % len(colors)]
            label_suffix = f"({yi},{xi})"

            spec_folded = folded.folded_spectrum.isel(polarity=i_pol, y=yi, x=xi).values
            spec_anti = folded.antisymmetric_spectrum.isel(polarity=i_pol, y=yi, x=xi).values

            spec_low = spec_folded + spec_anti / 2.0
            spec_high = spec_folded - spec_anti / 2.0

            ax.plot(
                delta_f_mhz, spec_low, color=color, lw=0.8, alpha=0.5, label=f"S_low {label_suffix}"
            )
            ax.plot(
                delta_f_mhz,
                spec_high,
                color=color,
                lw=0.8,
                alpha=0.5,
                ls="--",
                label=f"S_high {label_suffix}",
            )
            ax.plot(delta_f_mhz, spec_folded, color=color, lw=1.5, label=f"folded {label_suffix}")
            ax2.plot(
                delta_f_mhz,
                spec_anti,
                color=color,
                lw=1.0,
                alpha=0.7,
                ls=":",
                label=f"anti {label_suffix}",
            )

        ax2.axhline(0, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel("delta_f (MHz)")
        ax.set_ylabel("Intensity")
        ax2.set_ylabel("Antisymmetric", color="0.4")
        ax2.tick_params(axis="y", labelcolor="0.4")

        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, fontsize=7, loc="lower right")
        ax.set_title(f"Pixel spectra -- pol={pol_labels[i_pol]}")

    n_pixels = len(pixels)
    pixel_str = ", ".join(f"({yi},{xi})" for yi, xi in pixels)
    plt.suptitle(
        f"Folded ODMR spectra -- {n_pixels} pixel{'s' if n_pixels != 1 else ''}: {pixel_str}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_folding_mean_spectrum(folded: FoldedODMR) -> None:
    """Plot the spatially-averaged folded, unfolded, and antisymmetric spectra.

    One subplot per polarity showing:
    - Mean folded spectrum (symmetric average, should show a clear ODMR dip)
    - Mean S_low and S_high halves (recovered from folded +/- anti/2)
    - Mean antisymmetric component on a secondary y-axis (should be near zero)

    Args:
        folded: FoldedODMR result.
    """
    delta_f = folded.folded_spectrum.coords["delta_f_ghz"].values  # (n_df,)
    delta_f_mhz = delta_f * 1000

    pol_labels = list(folded.folded_spectrum.coords["polarity"].values)
    n_pol = len(pol_labels)

    _fig, axes = plt.subplots(1, n_pol, figsize=(6 * n_pol, 4), squeeze=False)

    for i_pol in range(n_pol):
        ax = axes[0, i_pol]

        mean_folded = folded.folded_spectrum.isel(polarity=i_pol).values.mean(axis=(0, 1))
        mean_anti = folded.antisymmetric_spectrum.isel(polarity=i_pol).values.mean(axis=(0, 1))

        # Recover individual halves: folded = (low+high)/2, anti = low-high
        mean_low = mean_folded + mean_anti / 2.0
        mean_high = mean_folded - mean_anti / 2.0

        ax.plot(delta_f_mhz, mean_low, "tab:cyan", lw=1, alpha=0.6, label="S_low(D-df)")
        ax.plot(delta_f_mhz, mean_high, "tab:orange", lw=1, alpha=0.6, label="S_high(D+df)")
        ax.plot(delta_f_mhz, mean_folded, "tab:blue", lw=1.5, label="folded (mean)")
        ax.set_xlabel("delta_f (MHz)")
        ax.set_ylabel("Mean intensity")

        ax2 = ax.twinx()
        ax2.plot(delta_f_mhz, mean_anti, "tab:red", lw=1, alpha=0.7, label="antisymmetric")
        ax2.axhline(0, color="0.5", ls=":", lw=0.8)
        ax2.set_ylabel("Antisymmetric", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Combined legend
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, fontsize=8, loc="lower right")

        ax.set_title(f"Mean folded spectrum -- pol={pol_labels[i_pol]}")

    plt.suptitle("Mean folded ODMR spectrum", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_folding_overview(folded: FoldedODMR) -> None:
    """2x2 diagnostic overview of the folding result.

    Panels:
    - (0, 0): Search landscape pol=0
    - (0, 1): Search landscape pol=1
    - (1, 0): D_ZFS spatial map (mean of polarities, deviation from nominal)
    - (1, 1): Fold residual spatial map (mean of polarities)

    Args:
        folded: FoldedODMR result.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    pol_labels = list(folded.d_zfs_map.coords["polarity"].values)
    n_pol = len(pol_labels)

    # Row 0: Search landscape
    if folded.d_candidates is not None and folded.search_residual is not None:
        d_cand = folded.d_candidates
        residual = folded.search_residual

        for i_pol in range(min(n_pol, 2)):
            ax = axes[0, i_pol]
            res = residual[i_pol]
            valid = np.isfinite(res)
            ax.plot(d_cand[valid], res[valid], "k-", lw=1)

            best_idx = np.argmin(res)
            ax.axvline(
                d_cand[best_idx],
                color="tab:red",
                ls="--",
                lw=1.5,
                label=f"min @ {d_cand[best_idx]:.6f}",
            )
            ax.set_xlabel("D candidate (GHz)")
            ax.set_ylabel("Mean residual")
            ax.set_title(f"Search -- pol={pol_labels[i_pol]}")
            ax.legend(fontsize=7)
    else:
        for i in range(2):
            axes[0, i].text(
                0.5,
                0.5,
                "No search data",
                transform=axes[0, i].transAxes,
                ha="center",
                va="center",
            )

    # Row 1, col 0: D_ZFS deviation map (mean over polarities)
    d_map = folded.d_zfs_map.values  # (n_pol, y, x)
    d_mean = np.mean(d_map, axis=0)  # (y, x)
    im0 = axes[1, 0].imshow(
        (d_mean - D_ZFS) * 1000,
        cmap="RdBu_r",
        origin="upper",
        aspect="equal",
    )
    axes[1, 0].set_title("D_ZFS deviation (MHz)")
    fig.colorbar(im0, ax=axes[1, 0], label="dD (MHz)")

    # Row 1, col 1: Fold residual map (mean over polarities)
    res_map = folded.fold_residual.values  # (n_pol, y, x)
    res_mean = np.mean(res_map, axis=0)  # (y, x)
    im1 = axes[1, 1].imshow(
        res_mean,
        cmap="magma",
        origin="upper",
        aspect="equal",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("Fold residual (0=good)")
    fig.colorbar(im1, ax=axes[1, 1], label="residual")

    plt.suptitle("Spectral folding diagnostics", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# ODMR spectrum plots
# ---------------------------------------------------------------------------


def plot_odmr_spectra(odmr_data: ODMRData, y: int, x: int) -> None:
    """Plot all ODMR spectra for pixel (y, x) in a polarity x freq_range grid.

    Each subplot shows one (polarity, freq_range) combination.

    Args:
        odmr_data: ODMRData instance to plot from.
        y: Row index in the scan grid.
        x: Column index in the scan grid.
    """
    da = odmr_data.data
    polarities = da.coords["polarity"].values.tolist()
    freq_ranges = da.coords["freq_range"].values.tolist()

    n_pol = len(polarities)
    n_frange = len(freq_ranges)
    fig, axes = plt.subplots(n_pol, n_frange, figsize=(5 * n_frange, 3 * n_pol), squeeze=False)

    for i, pol in enumerate(polarities):
        for j, fr in enumerate(freq_ranges):
            freq = da.coords["freq_ghz"].sel(freq_range=fr).values
            intensity = da.sel(polarity=pol, freq_range=fr).values[y, x, :]
            axes[i, j].plot(freq, intensity)
            axes[i, j].set_title(f"polarity={pol}, freq_range={fr}")
            axes[i, j].set_xlabel("Frequency (GHz)")
            axes[i, j].set_ylabel("Intensity")

    fig.suptitle(f"ODMR spectra at pixel ({y}, {x})")
    plt.tight_layout()
    plt.show()


def plot_fluorescence_correction(
    odmr_data: ODMRData,
    correction_factor: float = 0.2,
    pixel_idx: int | None = None,
) -> None:
    """Preview the effect of fluorescence correction on ODMR data.

    Args:
        odmr_data: ODMRData instance.
        correction_factor: The fluorescence correction factor.
        pixel_idx: Optional pixel index (flat y*x space) to highlight.
    """
    from qdmpy.odmr.processors import analyze_fluorescence_effects

    idx_flat, baseline_corrected = analyze_fluorescence_effects(odmr_data, pixel_idx)
    correction = correction_factor * baseline_corrected

    n_pol = odmr_data.data.sizes["polarity"]
    n_frange = odmr_data.data.sizes["freq_range"]
    n_y = odmr_data.data.sizes["y"]
    n_x = odmr_data.data.sizes["x"]

    flat_values = odmr_data.data.values.reshape(n_pol, n_frange, n_y * n_x, -1)

    _f, ax = plt.subplots(
        n_pol,
        n_frange,
        sharex=False,
        sharey=True,
        figsize=(4 * n_frange, 3 * n_pol),
    )

    if n_pol == 1 and n_frange == 1:
        ax = np.array([[ax]])
    elif n_pol == 1:
        ax = np.array([ax])
    elif n_frange == 1:
        ax = np.array([ax]).T

    freq_ghz = odmr_data.data.coords["freq_ghz"].values

    for p in range(n_pol):
        for fr in range(n_frange):
            current_data = flat_values[p, fr, idx_flat].copy()
            freqs = freq_ghz[fr]
            corr_vals = correction.isel(polarity=p, freq_range=fr).values

            ax[p, fr].plot(freqs, current_data, "k.-", label="Original")
            ax[p, fr].plot(
                freqs,
                current_data - corr_vals,
                "r.-",
                label=f"Corrected (Factor={correction_factor})",
            )
            ax[p, fr].plot(freqs, 1 + corr_vals, "r--", alpha=0.5, label="Correction")

            polarity_label = {0: "+", 1: "-"}.get(p, f"P{p}")
            frange_label = {0: "Low", 1: "High"}.get(fr, f"F{fr}")
            ax[p, fr].set_title(f"Polarity: {polarity_label}, Frequency Range: {frange_label}")
            ax[p, fr].set_xlabel("Frequency [GHz]")
            ax[p, fr].set_ylabel("ODMR Contrast")
            ax[p, fr].legend()
            ax[p, fr].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f"Fluorescence Correction Preview (Pixel {idx_flat})", y=1.02)
    plt.show()


# ---------------------------------------------------------------------------
# Model detection plots
# ---------------------------------------------------------------------------


def plot_model_detection(spectra_4d: NDArray, freq: NDArray | None = None) -> None:
    """Plot the median spectra used for model detection with detected peaks marked.

    Useful for visually verifying the auto-detection result, especially when
    doubt is flagged.

    Args:
        spectra_4d: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).
        freq: Optional 2D frequency array (n_frange, n_freq) in GHz. If None,
              frequency index is used on the x-axis.
    """
    from qdmpy.fitting.guess import _relative_prominence, validate_array

    validate_array(spectra_4d, 4, "spectra_4d")
    n_pol, n_frange = spectra_4d.shape[0], spectra_4d.shape[1]
    median_data = np.median(spectra_4d, axis=2)  # (n_pol, n_frange, n_freq)

    fig, axes = plt.subplots(
        n_pol,
        n_frange,
        figsize=(4 * n_frange, 3 * n_pol),
        squeeze=False,
        sharex="col",
    )
    fig.suptitle("Model detection: median spectra with detected peaks", fontsize=12)

    for p, f in np.ndindex(n_pol, n_frange):
        ax = axes[p, f]
        spectrum = median_data[p, f]
        prominence = _relative_prominence(spectrum)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(-spectrum, prominence=prominence)

        x = freq[f] if freq is not None else np.arange(len(spectrum))
        x_label = "Frequency (GHz)" if freq is not None else "Frequency index"

        ax.plot(x, spectrum, color="steelblue", linewidth=1.2)
        if len(peaks):
            ax.plot(x[peaks], spectrum[peaks], "rv", markersize=8, label=f"{len(peaks)} peaks")
        ax.axhline(
            spectrum.max() - prominence,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label=f"threshold ({prominence:.5f})",
        )
        ax.set_title(f"pol={p}, frange={f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Magnetic map plots
# ---------------------------------------------------------------------------


def plot_magnetic_component(
    mag_map: MagneticMap,
    component: str = "Bz",
    **imshow_kwargs: object,
) -> None:
    """Quick matplotlib display of one MagneticMap component.

    Args:
        mag_map: MagneticMap instance.
        component: Which component to display (case-insensitive for Bx/By/Bz).
        **imshow_kwargs: Passed to xarray ``.plot(**imshow_kwargs)``.

    Raises:
        ValueError: If component is not recognized.
    """
    component_lower = component.lower()
    valid_components = {"b111", "bx", "by", "bz", "btotal"}

    if component_lower not in valid_components:
        raise ValueError(f"Component {component!r} not in {valid_components}")

    da = getattr(mag_map, component_lower)
    da.plot(**imshow_kwargs)
    plt.title(component)
    plt.show()


# ---------------------------------------------------------------------------
# B111 and measurement display plots
# ---------------------------------------------------------------------------


def _label_spatial_axes(ax: MplAxes) -> None:
    """Add standard x/y µm labels to a spatial-map axes."""
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")


def _avg_param_map(arr: NDArray, h: int, w: int) -> NDArray:
    """Reshape a parameter array to (h, w) by averaging over leading dims.

    Args:
        arr: Parameter array with shape (..., n_pixel) or (h, w).
        h: Spatial height.
        w: Spatial width.

    Returns:
        2-D array with shape (h, w).
    """
    n_pixel = h * w
    if arr.ndim == 1:
        return arr.reshape(h, w)
    return np.nanmean(arr.reshape(-1, n_pixel), axis=0).reshape(h, w)


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
    valid = {"remanent", "induced"}
    if component not in valid:
        msg = f"component must be one of {valid!r}, got {component!r}"
        raise ValueError(msg)

    b_map = result.b111[component].values  # (H, W), µT

    pixel_spacing_um = result.pixel_spacing * 1e6
    height, width = result.scan_dimensions
    extent = (0, width * pixel_spacing_um, 0, height * pixel_spacing_um)

    vmax = float(np.nanpercentile(np.abs(b_map), 99))

    _fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        b_map,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-vmax,
        vmax=vmax,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"B111 {component} (µT)")
    ax.set_title(f"B111 {component} ({result.model_name})")
    _label_spatial_axes(ax)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"b111_{component}_{result.model_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_measurement_images(measurement: Measurement) -> None:
    """Plot the light and laser optical images from a Measurement.

    Args:
        measurement: Measurement instance containing ``light_image`` and
            ``laser_image`` arrays.
    """
    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(measurement.light_image, cmap="gray", origin="upper", aspect="equal")
    axes[0].set_title("Light image")
    axes[0].axis("off")

    axes[1].imshow(measurement.laser_image, cmap="gray", origin="upper", aspect="equal")
    axes[1].set_title("Laser image")
    axes[1].axis("off")

    plt.suptitle("Optical images", fontsize=12)
    plt.tight_layout()
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
    except Exception as exc:  # pylint: disable=broad-except
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
        return

    _, width = fit_result.scan_dimensions

    b111_flat = fit_result.b111_remanent.flatten()
    percs = np.linspace(10, 90, n_sample_pixels)
    flat_indices = [int(np.argmin(np.abs(b111_flat - np.percentile(b111_flat, p)))) for p in percs]

    model: Model | None = None
    try:
        from qdmpy.fitting.models import ModelRegistry

        model = ModelRegistry.get(fit_result.model_name)
    except Exception as exc:  # pylint: disable=broad-except
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


def _draw_b111_row(
    fig: matplotlib.figure.Figure,
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
        fig: Parent figure for colorbar attachment.
        axes: 2-D axes grid.
        fit_result: FitResult providing b111 and chi2.
        extent: imshow extent tuple (x0, x1, y0, y1) in µm.
        height: Scan height in pixels.
        width: Scan width in pixels.
    """
    try:
        b_rem = fit_result.b111_remanent  # (height, width), µT
        b_ind = fit_result.b111_induced
        vmax_rem = float(np.nanpercentile(np.abs(b_rem), 99))
        im = axes[0, 0].imshow(
            b_rem,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-vmax_rem,
            vmax=vmax_rem,
        )
        axes[0, 0].set_title("B111 remanent (µT)")
        _label_spatial_axes(axes[0, 0])
        fig.colorbar(im, ax=axes[0, 0])
        vmax_ind = float(np.nanpercentile(np.abs(b_ind), 99))
        im = axes[0, 1].imshow(
            b_ind,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-vmax_ind,
            vmax=vmax_ind,
        )
        axes[0, 1].set_title("B111 induced (µT)")
        _label_spatial_axes(axes[0, 1])
        fig.colorbar(im, ax=axes[0, 1])
    except Exception:  # pylint: disable=broad-except
        # Single-polarity data: show mean centre map as fallback
        center_fb = _avg_param_map(fit_result.centers, height, width)
        im = axes[0, 0].imshow(
            center_fb, extent=extent, origin="lower", cmap="viridis", aspect="equal"
        )
        axes[0, 0].set_title("Centre (GHz)")
        _label_spatial_axes(axes[0, 0])
        fig.colorbar(im, ax=axes[0, 0])
        axes[0, 1].set_visible(False)

    chi2_map = _avg_param_map(fit_result.chi2, height, width)
    im = axes[0, 2].imshow(chi2_map, extent=extent, origin="lower", cmap="magma", aspect="equal")
    axes[0, 2].set_title("Chi-squared")
    _label_spatial_axes(axes[0, 2])
    fig.colorbar(im, ax=axes[0, 2])


def _draw_param_row(
    fig: matplotlib.figure.Figure,
    axes: NDArray,
    fit_result: FitResult,
    extent: tuple[float, float, float, float],
    height: int,
    width: int,
) -> None:
    """Draw mean centre, contrast, and linewidth maps into row 1.

    Args:
        fig: Parent figure for colorbar attachment.
        axes: 2-D axes grid.
        fit_result: FitResult providing centres, contrasts, and linewidths.
        extent: imshow extent tuple in µm.
        height: Scan height in pixels.
        width: Scan width in pixels.
    """
    center_map = _avg_param_map(fit_result.centers, height, width)
    im = axes[1, 0].imshow(
        center_map, extent=extent, origin="lower", cmap="viridis", aspect="equal"
    )
    axes[1, 0].set_title("Centre (GHz, mean)")
    _label_spatial_axes(axes[1, 0])
    fig.colorbar(im, ax=axes[1, 0])

    try:
        contrast_map = _avg_param_map(fit_result.contrasts, height, width)
    except Exception:  # pylint: disable=broad-except
        contrast_map = np.zeros((height, width))
    im = axes[1, 1].imshow(
        contrast_map, extent=extent, origin="lower", cmap="viridis", aspect="equal"
    )
    axes[1, 1].set_title("Contrast (mean)")
    _label_spatial_axes(axes[1, 1])
    fig.colorbar(im, ax=axes[1, 1])

    try:
        lw_map = _avg_param_map(fit_result.linewidths, height, width)
    except Exception:  # pylint: disable=broad-except
        lw_map = np.zeros((height, width))
    im = axes[1, 2].imshow(lw_map, extent=extent, origin="lower", cmap="viridis", aspect="equal")
    axes[1, 2].set_title("Linewidth (GHz, mean)")
    _label_spatial_axes(axes[1, 2])
    fig.colorbar(im, ax=axes[1, 2])


def plot_qdm_display(
    result: FitResult | QDMResult,
    measurement: Measurement | None = None,
    n_sample_pixels: int = 3,
) -> None:
    """Comprehensive overview display for a QDM fit result.

    Always shown:
      - B111 remanent and induced maps (µT, diverging colormap)
      - Chi-squared map
      - Mean resonance centre, contrast, and linewidth maps

    Shown when *measurement* is provided:
      - Light and laser optical images
      - ``n_sample_pixels`` representative pixel spectra with fit curves

    Pixels for spectral display are chosen at equally-spaced percentile
    positions of B111 remanent so they sample the full dynamic range.

    Args:
        result: FitResult or QDMResult.
        measurement: Optional Measurement for optical images and ODMR spectra.
        n_sample_pixels: Number of sample pixel spectra (default 3).
    """
    fit_result: FitResult = (  # type: ignore[assignment]
        result.fit_result if hasattr(result, "fit_result") else result  # type: ignore[union-attr]
    )

    height, width = fit_result.scan_dimensions
    pixel_spacing_um = fit_result.pixel_spacing * 1e6
    extent = (0, width * pixel_spacing_um, 0, height * pixel_spacing_um)

    spec_rows = -(-n_sample_pixels // 3) if measurement is not None else 0  # ceil div
    n_rows = 2 + (1 if measurement is not None else 0) + spec_rows
    n_cols = 3

    fig, axes_raw = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes: NDArray = np.atleast_2d(axes_raw)  # type: ignore[arg-type]

    _draw_b111_row(fig, axes, fit_result, extent, height, width)
    _draw_param_row(fig, axes, fit_result, extent, height, width)

    if measurement is not None:
        row = 2
        axes[row, 0].imshow(measurement.light_image, cmap="gray", origin="upper", aspect="equal")
        axes[row, 0].set_title("Light image")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(measurement.laser_image, cmap="gray", origin="upper", aspect="equal")
        axes[row, 1].set_title("Laser image")
        axes[row, 1].axis("off")
        axes[row, 2].set_visible(False)
        _plot_display_pixel_spectra(axes, row + 1, n_sample_pixels, fit_result, measurement, n_cols)

    plt.suptitle(f"QDM Result Overview ({fit_result.model_name})", fontsize=14)
    plt.tight_layout()
    plt.show()


__all__ = [
    "double_norm",
    "plot_b111_map",
    "plot_fit_result_field_map",
    "plot_fit_result_overview",
    "plot_fit_result_parameter_map",
    "plot_fluorescence_correction",
    "plot_folding_mean_spectrum",
    "plot_folding_overview",
    "plot_folding_pixel_spectra",
    "plot_folding_search_landscape",
    "plot_magnetic_component",
    "plot_measurement_images",
    "plot_model_detection",
    "plot_odmr_spectra",
    "plot_qdm_display",
    "resolve_pixel_indices",
]
