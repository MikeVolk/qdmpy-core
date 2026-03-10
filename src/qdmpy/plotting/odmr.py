"""ODMR spectra, folding diagnostics, and model-detection plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from qdmpy.constants import D_ZFS
from qdmpy.plotting._common import _add_colorbar, _finalize_layout, resolve_pixel_indices

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from qdmpy.odmr.data import ODMRData
    from qdmpy.odmr.folding import FoldedODMR


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

    logger.debug("Plotting folding search landscape")
    d_cand = folded.d_candidates
    residual = folded.search_residual  # (n_pol, n_steps)
    n_pol = residual.shape[0]
    pol_labels = list(folded.d_zfs_map.coords["polarity"].values)

    fig, axes = plt.subplots(1, n_pol, figsize=(5 * n_pol, 4), squeeze=False)

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

    fig.suptitle("D_ZFS brute-force search", fontsize=12)
    _finalize_layout(fig, reserve_top=0.05)
    plt.show()


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
    logger.debug("Plotting folding pixel spectra ({} pixels)", len(pixels))

    delta_f = folded.folded_spectrum.coords["delta_f_ghz"].values  # (n_df,)
    delta_f_mhz = delta_f * 1000

    pol_labels = list(folded.folded_spectrum.coords["polarity"].values)
    n_pol = len(pol_labels)

    fig, axes = plt.subplots(1, n_pol, figsize=(6 * n_pol, 4), squeeze=False)
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
    fig.suptitle(
        f"Folded ODMR spectra -- {n_pixels} pixel{'s' if n_pixels != 1 else ''}: {pixel_str}",
        fontsize=12,
    )
    _finalize_layout(fig, reserve_top=0.05)
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
    logger.debug("Plotting folding mean spectrum")
    delta_f = folded.folded_spectrum.coords["delta_f_ghz"].values  # (n_df,)
    delta_f_mhz = delta_f * 1000

    pol_labels = list(folded.folded_spectrum.coords["polarity"].values)
    n_pol = len(pol_labels)

    fig, axes = plt.subplots(1, n_pol, figsize=(6 * n_pol, 4), squeeze=False)

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

    fig.suptitle("Mean folded ODMR spectrum", fontsize=12)
    _finalize_layout(fig, reserve_top=0.05)
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
    logger.debug("Plotting folding overview")
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
    _add_colorbar(im0, axes[1, 0], label="dD (MHz)")

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
    _add_colorbar(im1, axes[1, 1], label="residual")

    fig.suptitle("Spectral folding diagnostics", fontsize=13)
    _finalize_layout(fig, reserve_top=0.06)
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
    logger.debug("Plotting ODMR spectra at pixel ({}, {})", y, x)
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
    _finalize_layout(fig, reserve_top=0.05)
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
    logger.debug("Plotting fluorescence correction preview")
    from qdmpy.odmr.processors import analyze_fluorescence_effects

    idx_flat, baseline_corrected = analyze_fluorescence_effects(odmr_data, pixel_idx)
    correction = correction_factor * baseline_corrected

    n_pol = odmr_data.data.sizes["polarity"]
    n_frange = odmr_data.data.sizes["freq_range"]
    n_y = odmr_data.data.sizes["y"]
    n_x = odmr_data.data.sizes["x"]

    flat_values = odmr_data.data.values.reshape(n_pol, n_frange, n_y * n_x, -1)

    fig, ax = plt.subplots(
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

    fig.suptitle(f"Fluorescence Correction Preview (Pixel {idx_flat})", y=1.02)
    _finalize_layout(fig, reserve_top=0.08)
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
    logger.debug("Plotting model detection diagnostics")
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

    _finalize_layout(fig, reserve_top=0.05)
    plt.show()
