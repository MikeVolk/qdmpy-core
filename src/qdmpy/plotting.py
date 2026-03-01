"""Visualization module for QDMpy.

This module provides plotting functions for visualizing data from Quantum Diamond
Microscopy (QDM) measurements, including magnetic field maps and spatial parameter maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from qdmpy.constants import D_ZFS
from qdmpy.utils import double_norm

if TYPE_CHECKING:
    from qdmpy.fitting.result import FitResult
    from qdmpy.odmr.folding import FoldedODMR

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


__all__ = [
    "double_norm",
    "plot_fit_result_field_map",
    "plot_fit_result_overview",
    "plot_fit_result_parameter_map",
    "plot_folding_mean_spectrum",
    "plot_folding_overview",
    "plot_folding_search_landscape",
]
