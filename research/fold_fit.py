# ruff: noqa
"""Fold-symmetrize ODMR spectra then GPU-fit and compare B111 maps.

Pipeline
--------
1. Load real data (MIL2_FOV1, 4×4 binned).
2. Find mean-spectrum fold centre per (polarity, freq_range) — 4 values.
3. Symmetrize each pixel's spectrum around that global centre:
       S_sym(f) = ( S(f) + S(2·f_c − f) ) / 2
   Same frequency axis, same shape — feeds directly into FitManager.
   Noise is reduced ~√2 because each dip is averaged with its mirror image.
4. GPU-fit both the original and the symmetrized spectra (same model).
5. Compute B111 remanent/induced from each fit.
6. Plot 2×3 comparison: (normal, fold-fit, difference) × (remanent, induced).

Output
------
All figures saved to research/output/fold_fit/.

Usage
-----
    uv run python research/fold_fit.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from qdmpy_core.fitting.manager import FitManager
from qdmpy_core.measurement import Measurement
from qdmpy_core.odmr.data import ODMRData

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("tests/data/MIL2_FOV1")
BIN_FACTOR = 4
MODEL = "auto"          # let qdmpy detect ESR14N / ESR15N / ESRSINGLE
N_FOLD_CANDIDATES = 500  # resolution of the fold-centre sweep

OUT_DIR = Path("research/output/fold_fit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 — fold centre from the mean spectrum
# ---------------------------------------------------------------------------
def _fold_residuals(spectrum: NDArray, freqs: NDArray, centers: NDArray) -> NDArray:
    """Return the fold residual for every candidate centre (vectorized over centers)."""
    step = float(np.median(np.diff(freqs)))
    residuals = np.empty(len(centers))
    for i, c in enumerate(centers):
        reach = min(c - freqs[0], freqs[-1] - c)
        offsets = np.arange(step, reach + step / 2, step)
        if len(offsets) == 0:
            residuals[i] = np.inf
            continue
        left = np.interp(c - offsets, freqs, spectrum)
        right = np.interp(c + offsets, freqs, spectrum)
        residuals[i] = float(np.sum((left - right) ** 2))
    return residuals


def find_fold_center(spectrum: NDArray, freqs: NDArray) -> float:
    """Find the fold centre that best symmetrises *spectrum*."""
    margin = 0.2 * (freqs[-1] - freqs[0])
    centers = np.linspace(freqs[0] + margin, freqs[-1] - margin, N_FOLD_CANDIDATES)
    res = _fold_residuals(spectrum, freqs, centers)
    return float(centers[np.argmin(res)])


def mean_fold_centers(data: xr.DataArray) -> dict[str, dict[str, float]]:
    """Compute fold centre from the mean spectrum for every (pol, frange).

    Returns:
        Nested dict: centers[pol][frange] = fold_centre_ghz
    """
    freq_ghz = data.coords["freq_ghz"].values         # (n_frange, n_freq)
    frange_labels = list(data.coords["freq_range"].values)
    pol_labels = list(data.coords["polarity"].values)

    centers: dict[str, dict[str, float]] = {}
    for pol in pol_labels:
        centers[pol] = {}
        for fi, frange in enumerate(frange_labels):
            mean_spec = data.sel(polarity=pol, freq_range=frange).values.mean(axis=(0, 1))
            fc = find_fold_center(mean_spec, freq_ghz[fi])
            centers[pol][frange] = fc
            print(f"  fold centre  pol={pol}  frange={frange}:  {fc:.6f} GHz")
    return centers


# ---------------------------------------------------------------------------
# Step 2 — symmetrize spectra around per-(pol, frange) fold centres
# ---------------------------------------------------------------------------
def _symmetrize_band(
    spectra: NDArray,
    freqs: NDArray,
    f_c: float,
) -> NDArray:
    """Symmetrize all pixel spectra around *f_c* on the original frequency axis.

    S_sym(f) = ( S(f) + S(2·f_c − f) ) / 2

    Args:
        spectra: (ny, nx, n_freq) float array.
        freqs: (n_freq,) GHz array, monotonically increasing.
        f_c: Fold centre in GHz.

    Returns:
        Symmetrized spectra, same shape as *spectra*.
    """
    ny, nx, n_freq = spectra.shape
    mirror_freqs = 2.0 * f_c - freqs           # (n_freq,) — mirrored query points

    # Clamp mirror queries to [freqs[0], freqs[-1]] for interp
    mirror_freqs_clamped = np.clip(mirror_freqs, freqs[0], freqs[-1])

    # Fractional index positions on the original freq axis (same for all pixels)
    idx_float = np.interp(mirror_freqs_clamped, freqs, np.arange(n_freq))
    idx_lo = np.clip(np.floor(idx_float).astype(int), 0, n_freq - 2)
    frac = idx_float - idx_lo                  # (n_freq,)

    # Gather mirror values for all pixels at once
    flat = spectra.reshape(-1, n_freq)         # (n_pix, n_freq)
    # flat[:, idx_lo] * (1 - frac) + flat[:, idx_lo+1] * frac
    mirror_vals = flat[:, idx_lo] * (1.0 - frac) + flat[:, idx_lo + 1] * frac
    mirror_vals = mirror_vals.reshape(ny, nx, n_freq)

    return (spectra + mirror_vals) / 2.0


def symmetrize_data(
    data: xr.DataArray,
    fold_centers: dict[str, dict[str, float]],
) -> xr.DataArray:
    """Return a new DataArray with each band symmetrized around its fold centre.

    The returned array has identical dims, coords, and shape to *data*.
    """
    arr = data.values.copy()                   # (n_pol, n_frange, ny, nx, n_freq)
    freq_ghz = data.coords["freq_ghz"].values  # (n_frange, n_freq)
    pol_labels = list(data.coords["polarity"].values)
    frange_labels = list(data.coords["freq_range"].values)

    for pi, pol in enumerate(pol_labels):
        for fi, frange in enumerate(frange_labels):
            f_c = fold_centers[pol][frange]
            arr[pi, fi] = _symmetrize_band(arr[pi, fi], freq_ghz[fi], f_c)

    return xr.DataArray(arr, dims=data.dims, coords=data.coords)


# ---------------------------------------------------------------------------
# Step 3 — GPU fit (shared helper)
# ---------------------------------------------------------------------------
def run_fit(odmr_data: ODMRData, label: str) -> tuple[NDArray, NDArray]:
    """GPU-fit *odmr_data* and return (b111_remanent, b111_induced) in µT."""
    fm = FitManager(model_name=MODEL)
    t0 = time.perf_counter()
    result = fm.fit(
        data=odmr_data.data,
        frequencies=odmr_data.frequencies,
        pixel_spacing=1.0,          # spacing not needed for B111 calculation
    )
    dt = time.perf_counter() - t0
    print(f"  [{label}] fit completed in {dt:.1f}s")
    return result.b111_remanent, result.b111_induced


# ---------------------------------------------------------------------------
# Step 4 — comparison plots
# ---------------------------------------------------------------------------
def save_comparison(
    normal: dict[str, NDArray],
    fold: dict[str, NDArray],
    pixel_spacing_um: float,
    label_a: str = "normal fit",
    label_b: str = "fold-symmetrized fit",
) -> None:
    """2×3 grid: (remanent, induced) × (A, B, A−B)."""
    keys = ("remanent", "induced")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"B₁₁₁ comparison: {label_a} vs {label_b}", fontsize=12)

    for row, key in enumerate(keys):
        a = normal[key]
        b = fold[key]
        diff = b - a

        vmin = float(np.percentile(np.concatenate([a.ravel(), b.ravel()]), 1))
        vmax = float(np.percentile(np.concatenate([a.ravel(), b.ravel()]), 99))
        dv = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)))

        ny, nx = a.shape
        ext = [0, nx * pixel_spacing_um, ny * pixel_spacing_um, 0]

        im0 = axes[row, 0].imshow(a, extent=ext, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[row, 0].set_title(f"B₁₁₁ {key}\n{label_a}")

        im1 = axes[row, 1].imshow(b, extent=ext, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[row, 1].set_title(f"B₁₁₁ {key}\n{label_b}")

        im2 = axes[row, 2].imshow(diff, extent=ext, vmin=-dv, vmax=dv, cmap="RdBu_r")
        axes[row, 2].set_title(f"diff ({label_b} − {label_a})\nRMS={np.sqrt(np.mean(diff**2)):.2f} µT")

        for ax, im in [(axes[row, 0], im0), (axes[row, 1], im1), (axes[row, 2], im2)]:
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            fig.colorbar(im, ax=ax, label="µT", shrink=0.8)

    fig.tight_layout()
    path = OUT_DIR / "b111_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")


def save_fold_diagnostic(
    original: xr.DataArray,
    symmetrized: xr.DataArray,
    fold_centers: dict[str, dict[str, float]],
) -> None:
    """4-panel plot showing one pixel's original vs symmetrized spectrum per band."""
    freq_ghz = original.coords["freq_ghz"].values
    pol_labels = list(original.coords["polarity"].values)
    frange_labels = list(original.coords["freq_range"].values)

    mid_y = original.sizes["y"] // 2
    mid_x = original.sizes["x"] // 2

    fig, axes = plt.subplots(
        len(pol_labels), len(frange_labels),
        figsize=(5 * len(frange_labels), 4 * len(pol_labels)),
        squeeze=False,
    )
    fig.suptitle(f"Original vs fold-symmetrized spectrum — pixel ({mid_y}, {mid_x})", fontsize=11)

    for pi, pol in enumerate(pol_labels):
        for fi, frange in enumerate(frange_labels):
            ax = axes[pi, fi]
            freqs = freq_ghz[fi]
            f_c = fold_centers[pol][frange]

            orig_spec = original.sel(polarity=pol, freq_range=frange).values[mid_y, mid_x, :]
            sym_spec = symmetrized.sel(polarity=pol, freq_range=frange).values[mid_y, mid_x, :]

            ax.plot(freqs, orig_spec, "0.6", lw=1.0, label="original")
            ax.plot(freqs, sym_spec, "tab:blue", lw=1.5, label="symmetrized")
            ax.axvline(f_c, color="tab:red", ls="--", lw=1.0, label=f"fold centre {f_c:.5f}")
            ax.set_title(f"pol={pol}  frange={frange}", fontsize=9)
            ax.set_xlabel("Frequency (GHz)", fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.legend(fontsize=7)

    fig.tight_layout()
    path = OUT_DIR / "spectrum_diagnostic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")


def save_histograms(normal: dict[str, NDArray], fold: dict[str, NDArray]) -> None:
    """Overlay histograms of B111 values for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("B₁₁₁ value distributions: normal vs fold-symmetrized fit")

    for ax, key in zip(axes, ("remanent", "induced")):
        a, b = normal[key].ravel(), fold[key].ravel()
        lo = float(np.percentile(np.concatenate([a, b]), 0.5))
        hi = float(np.percentile(np.concatenate([a, b]), 99.5))
        bins = np.linspace(lo, hi, 80)
        ax.hist(a, bins=bins, alpha=0.6, label="normal fit", color="tab:blue")
        ax.hist(b, bins=bins, alpha=0.6, label="fold-symmetrized", color="tab:orange")
        ax.set_title(f"B₁₁₁ {key}")
        ax.set_xlabel("µT")
        ax.set_ylabel("pixel count")
        ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "b111_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {DATA_DIR} (bin={BIN_FACTOR}) …")
    m = Measurement.from_folder(DATA_DIR, bin_factor=BIN_FACTOR, normalize=True)
    processed = m.odmr.processed_data
    data = processed.data
    pixel_spacing_um = m.pixel_spacing * 1e6

    print(f"  shape: {data.shape}   pixel spacing: {pixel_spacing_um:.2f} µm\n")

    # ------------------------------------------------------------------
    # 1. Mean-spectrum fold centres
    # ------------------------------------------------------------------
    print("Step 1 — computing mean-spectrum fold centres …")
    fold_centers = mean_fold_centers(data)

    # ------------------------------------------------------------------
    # 2. Symmetrize spectra
    # ------------------------------------------------------------------
    print("\nStep 2 — symmetrizing spectra …")
    t0 = time.perf_counter()
    sym_data = symmetrize_data(data, fold_centers)
    print(f"  symmetrization took {time.perf_counter() - t0:.2f}s")

    sym_odmr = ODMRData(data=sym_data, metadata={"source": "fold_symmetrized"})

    # Diagnostic: show original vs symmetrized at centre pixel
    save_fold_diagnostic(data, sym_data, fold_centers)

    # ------------------------------------------------------------------
    # 3. GPU fits
    # ------------------------------------------------------------------
    print("\nStep 3 — GPU fitting …")
    print("  [a] fitting original spectra …")
    b111_normal_rem, b111_normal_ind = run_fit(processed, "normal")

    print("  [b] fitting fold-symmetrized spectra …")
    b111_fold_rem, b111_fold_ind = run_fit(sym_odmr, "fold-sym")

    # ------------------------------------------------------------------
    # 4. Summary statistics
    # ------------------------------------------------------------------
    print("\n--- B₁₁₁ summary ---")
    for key, arr_n, arr_f in [
        ("remanent", b111_normal_rem, b111_fold_rem),
        ("induced", b111_normal_ind, b111_fold_ind),
    ]:
        diff = arr_f - arr_n
        print(f"  {key}:")
        print(f"    normal : mean={arr_n.mean():.2f}  median={np.median(arr_n):.2f}  std={arr_n.std():.2f} µT")
        print(f"    fold   : mean={arr_f.mean():.2f}  median={np.median(arr_f):.2f}  std={arr_f.std():.2f} µT")
        print(f"    diff   : RMS={np.sqrt(np.mean(diff**2)):.2f}  max|Δ|={np.abs(diff).max():.2f} µT")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("\nStep 4 — saving plots …")
    normal = {"remanent": b111_normal_rem, "induced": b111_normal_ind}
    fold   = {"remanent": b111_fold_rem,   "induced": b111_fold_ind}

    save_comparison(normal, fold, pixel_spacing_um)
    save_histograms(normal, fold)

    print(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
