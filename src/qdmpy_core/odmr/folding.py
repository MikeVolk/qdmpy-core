"""Spectral folding for ODMR data (QEP-011).

Exploits the f+/f- mirror symmetry of NV-center ODMR spectra about D_ZFS:

    f+/- = D +/- gamma*B*cos(theta)

The low frequency range captures f- and the high range captures f+. Folding
them together about D gives sqrt(2) SNR improvement, a per-pixel D_ZFS map
(temperature/strain), and a model-free quality metric (fold residual).

Algorithm (two-scale):
    1. Spatially coarsen (bin_factor^2) -> high SNR for coarse D_ZFS estimation
    2. Brute-force search over D candidates -> coarse D_ZFS map
    3. Bicubic-interpolate coarse map to full resolution
    4. Per-pixel fold: S_folded(df) = (S_low(D-df) + S_high(D+df)) / 2
    5. Fold residual: measure of spectrum asymmetry (quality map)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import zoom

from qdmpy_core.constants import D_ZFS
from qdmpy_core.exceptions import DataValidationError, FoldingOverlapError

if TYPE_CHECKING:
    from qdmpy_core.odmr.data import ODMRData

_MIN_FREQ_RANGES = 2


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class FoldingSettings(BaseModel):
    """Immutable settings for the spectral folding pipeline.

    Attributes:
        d_zfs_initial: Starting centre for D_ZFS search in GHz.
        search_range: Half-width of brute-force search in GHz (+/-5 MHz = +/-68 K).
        search_steps: Number of candidate D values in the search grid.
        bin_factor: Spatial binning factor for coarse D_ZFS estimation.
        interpolation_order: scipy.ndimage.zoom order (3 = bicubic).
        min_overlap_points: Minimum df points required; raises FoldingOverlapError otherwise.
    """

    model_config = ConfigDict(frozen=True)

    d_zfs_initial: float = D_ZFS
    search_range: float = 0.005
    search_steps: int = 201
    bin_factor: int = 8
    interpolation_order: int = 3
    min_overlap_points: int = 5


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class FoldedODMR(BaseModel):
    """Result of spectral folding.

    Attributes:
        folded_spectrum: (polarity, y, x, freq_idx) with coord delta_f_ghz.
            Mean of low and high halves: (S_low(D-df) + S_high(D+df)) / 2.
        antisymmetric_spectrum: Same dims. Antisymmetric component S_low-S_high
            (quality diagnostic -- should be near zero for symmetric data).
        d_zfs_map: (polarity, y, x) per-pixel fold centre in GHz.
        fold_residual: (polarity, y, x) normalised [0, 1]; low = symmetric = good.
        settings: The FoldingSettings used to produce this result.
        d_candidates: 1D array of D search grid values in GHz (n_steps,).
        search_residual: (polarity, n_steps) mean residual per D candidate,
            averaged over all coarse super-pixels. Used for diagnostic plots.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    folded_spectrum: xr.DataArray
    antisymmetric_spectrum: xr.DataArray
    d_zfs_map: xr.DataArray
    fold_residual: xr.DataArray
    settings: FoldingSettings
    d_candidates: NDArray | None = None
    search_residual: NDArray | None = None

    def plot(self) -> None:
        """Quick diagnostic overview of the folding result."""
        from qdmpy_core.plotting import plot_folding_overview

        plot_folding_overview(self)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _overlap_range(
    f_low: NDArray,
    f_high: NDArray,
    d: float,
) -> tuple[float, float]:
    """Compute the valid Zeeman-offset (df) overlap range for a given D.

    The low range covers [f_low[0], f_low[-1]] and the high range covers
    [f_high[0], f_high[-1]]. For fold point D the valid df satisfies:

        D - df in [f_low[0], f_low[-1]] and D + df in [f_high[0], f_high[-1]]

    Args:
        f_low: 1D low-range frequency array in GHz.
        f_high: 1D high-range frequency array in GHz.
        d: Candidate fold point in GHz.

    Returns:
        (df_inner, df_outer) -- the valid df interval in GHz.

    Raises:
        FoldingOverlapError: If there is no valid overlap (df_outer <= df_inner).
    """
    df_inner = max(d - f_low[-1], f_high[0] - d)
    df_outer = min(d - f_low[0], f_high[-1] - d)
    if df_outer <= df_inner:
        msg = (
            f"No valid overlap for D={d:.6f} GHz: df_inner={df_inner:.6f}, df_outer={df_outer:.6f}"
        )
        raise FoldingOverlapError(msg)
    return df_inner, df_outer


def _interp_batch(
    spectra: NDArray,
    freqs: NDArray,
    query_freqs: NDArray,
) -> NDArray:
    """Vectorised linear interpolation without scipy dependency.

    Uses a fractional-index approach that assumes approximately uniform
    frequency spacing (typical for ODMR sweep data).

    Args:
        spectra: Shape (N, n_freq) -- batch of spectra to interpolate.
        freqs: Shape (n_freq,) -- frequency axis for all spectra.
        query_freqs: Shape (N, n_q) -- per-row query frequencies.

    Returns:
        Interpolated values, shape (N, n_q).
    """
    n_freq = len(freqs)
    step = float(np.median(np.diff(freqs)))

    # Fractional indices into freqs
    idx = (query_freqs - freqs[0]) / step  # (N, n_q)
    idx_lo = np.clip(np.floor(idx).astype(np.intp), 0, n_freq - 2)
    frac = np.clip(idx - idx_lo, 0.0, 1.0)

    # Advanced indexing: gather spectra values at idx_lo and idx_lo+1
    row_idx = np.arange(spectra.shape[0])[:, None]  # (N, 1)
    s_lo = spectra[row_idx, idx_lo]  # (N, n_q)
    s_hi = spectra[row_idx, idx_lo + 1]  # (N, n_q)

    return s_lo * (1.0 - frac) + s_hi * frac


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SpectralFolder:
    """Folds ODMR spectra about the per-pixel D_ZFS value.

    Usage::

        folder = SpectralFolder(odmr_data)
        result = folder.fold()
        # result.folded_spectrum  -- shape (polarity, y, x, freq_idx)
        # result.d_zfs_map        -- shape (polarity, y, x)
        # result.fold_residual    -- shape (polarity, y, x)
    """

    def __init__(
        self,
        odmr_data: ODMRData,
        settings: FoldingSettings | None = None,
    ) -> None:
        """Initialise SpectralFolder.

        Args:
            odmr_data: ODMR data with both 'low' and 'high' frequency ranges.
            settings: Folding configuration. Defaults to FoldingSettings().
        """
        self._odmr_data = odmr_data
        self._settings = settings if settings is not None else FoldingSettings()

    def fold(self) -> FoldedODMR:
        """Run full pipeline: coarsen -> D_ZFS map -> interpolate -> fold -> residual.

        Returns:
            FoldedODMR with folded spectra, D_ZFS map, antisymmetric component,
            and fold residual.

        Raises:
            DataValidationError: If the data does not have both 'low' and 'high'
                frequency ranges.
            FoldingOverlapError: If the df overlap window has fewer than
                min_overlap_points points, or if no D candidate in the search
                range satisfies the min_overlap_points threshold.
        """
        data = self._odmr_data.data

        if data.sizes.get("freq_range", 0) < _MIN_FREQ_RANGES:
            msg = (
                "SpectralFolder requires ODMRData with both 'low' and 'high' "
                "frequency ranges (freq_range dim must have size >= 2)."
            )
            raise DataValidationError(msg)

        logger.debug(
            "SpectralFolder.fold(): data shape={}, bin_factor={}, search_steps={}",
            data.shape,
            self._settings.bin_factor,
            self._settings.search_steps,
        )

        # Step 1: Coarsen
        coarse = self._coarsen_data()
        logger.debug(
            "Coarse shape: {} x {} (from {} x {})",
            coarse.sizes["y"],
            coarse.sizes["x"],
            data.sizes["y"],
            data.sizes["x"],
        )

        # Step 2: Find D_ZFS at coarse resolution
        d_zfs_coarse, d_candidates, search_residual = self._find_d_zfs_coarse(coarse)
        logger.debug(
            "Coarse D_ZFS: {:.6f} - {:.6f} GHz",
            float(d_zfs_coarse.min()),
            float(d_zfs_coarse.max()),
        )

        # Step 3: Interpolate to full resolution
        target_shape = (data.sizes["y"], data.sizes["x"])
        d_zfs_full = self._interpolate_d_zfs(d_zfs_coarse, target_shape)  # (n_pol, ny, nx)

        # Step 4: Fold spectra
        folded, antisymmetric, _delta_f = self._fold_spectra(d_zfs_full)

        # Step 5: Compute fold residual
        fold_residual = self._compute_fold_residual(antisymmetric, folded)

        # Wrap D_ZFS map
        pol_labels = list(data.coords["polarity"].values)
        d_zfs_da = xr.DataArray(
            d_zfs_full,
            dims=("polarity", "y", "x"),
            coords={"polarity": pol_labels},
        )

        return FoldedODMR(
            folded_spectrum=folded,
            antisymmetric_spectrum=antisymmetric,
            d_zfs_map=d_zfs_da,
            fold_residual=fold_residual,
            settings=self._settings,
            d_candidates=d_candidates,
            search_residual=search_residual,
        )

    # -----------------------------------------------------------------------
    # Private pipeline steps
    # -----------------------------------------------------------------------

    def _coarsen_data(self) -> xr.DataArray:
        """Coarsen ODMR data spatially by bin_factor for high-SNR D_ZFS search."""
        bf = self._settings.bin_factor
        return self._odmr_data.data.coarsen(y=bf, x=bf, boundary="trim").mean()  # type: ignore[attr-defined]

    def _find_d_zfs_coarse(
        self, coarse_data: xr.DataArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Brute-force D_ZFS search at coarse resolution.

        For each polarity and each spatial super-pixel, sweeps search_steps
        candidate D values and picks the one that best aligns S_low(D-df)
        with S_high(D+df).

        Returns:
            Tuple of (d_zfs_map, d_candidates, mean_residual):
            - d_zfs_map: (n_pol, ny_c, nx_c) with D_ZFS in GHz.
            - d_candidates: (n_steps,) search grid in GHz.
            - mean_residual: (n_pol, n_steps) mean residual per candidate,
              averaged over all coarse super-pixels.

        Raises:
            FoldingOverlapError: If no candidate D has a valid overlap window
                wide enough to satisfy min_overlap_points.
        """
        settings = self._settings
        freq_ghz = coarse_data.coords["freq_ghz"].values  # (2, n_freq)
        f_low = freq_ghz[0]
        f_high = freq_ghz[1]
        step = float(np.median(np.diff(f_low)))

        d_candidates = np.linspace(
            settings.d_zfs_initial - settings.search_range,
            settings.d_zfs_initial + settings.search_range,
            settings.search_steps,
        )

        n_pol = coarse_data.sizes["polarity"]
        ny_c = coarse_data.sizes["y"]
        nx_c = coarse_data.sizes["x"]
        result = np.empty((n_pol, ny_c, nx_c))
        mean_residual = np.full((n_pol, len(d_candidates)), np.inf)

        for i_pol in range(n_pol):
            spec_low = (
                coarse_data.isel(polarity=i_pol).sel(freq_range="low").values
            )  # (ny_c, nx_c, n_freq)
            spec_high = (
                coarse_data.isel(polarity=i_pol).sel(freq_range="high").values
            )  # (ny_c, nx_c, n_freq)

            flat_low = spec_low.reshape(-1, spec_low.shape[-1])  # (N, n_freq)
            flat_high = spec_high.reshape(-1, spec_high.shape[-1])  # (N, n_freq)
            n_pixels = flat_low.shape[0]

            residuals = np.full((len(d_candidates), n_pixels), np.inf)
            any_valid = False

            for j, d in enumerate(d_candidates):
                df_inner = max(d - f_low[-1], f_high[0] - d)
                df_outer = min(d - f_low[0], f_high[-1] - d)

                if df_outer - df_inner < settings.min_overlap_points * step:
                    continue

                any_valid = True
                delta_f = np.arange(df_inner, df_outer, step)

                # Fractional indices for all df points (same for every pixel)
                idx_l = (d - delta_f - f_low[0]) / step
                idx_h = (d + delta_f - f_high[0]) / step
                idx_lo_l = np.clip(np.floor(idx_l).astype(np.intp), 0, len(f_low) - 2)
                idx_lo_h = np.clip(np.floor(idx_h).astype(np.intp), 0, len(f_high) - 2)
                frac_l = np.clip(idx_l - idx_lo_l, 0.0, 1.0)
                frac_h = np.clip(idx_h - idx_lo_h, 0.0, 1.0)

                # Vectorised gather: (N, n_df)
                s_l = flat_low[:, idx_lo_l] * (1.0 - frac_l) + flat_low[:, idx_lo_l + 1] * frac_l
                s_h = flat_high[:, idx_lo_h] * (1.0 - frac_h) + flat_high[:, idx_lo_h + 1] * frac_h

                residuals[j] = ((s_l - s_h) ** 2).mean(axis=1)  # (N,)

            if not any_valid:
                msg = (
                    f"No overlap window met the min_overlap_points={settings.min_overlap_points} "
                    f"requirement for polarity={i_pol}. All {len(d_candidates)} D candidates had "
                    f"fewer than {settings.min_overlap_points} overlap points. "
                    f"Check search_range={settings.search_range:.4f} GHz or min_overlap_points."
                )
                raise FoldingOverlapError(msg)

            best_idx = np.argmin(residuals, axis=0)  # (N,)
            result[i_pol] = d_candidates[best_idx].reshape(ny_c, nx_c)
            mean_residual[i_pol] = np.mean(residuals, axis=1)  # (n_steps,)

        return result, d_candidates, mean_residual

    def _interpolate_d_zfs(
        self,
        d_zfs_coarse: NDArray,
        target_shape: tuple[int, int],
    ) -> NDArray:
        """Bicubic-interpolate coarse D_ZFS map to full resolution.

        Uses exact per-axis zoom factors so the output matches target_shape
        regardless of boundary='trim' truncation.

        Returns:
            NDArray of shape (n_pol, ny, nx).
        """
        n_pol = d_zfs_coarse.shape[0]
        ny_full, nx_full = target_shape
        result = np.empty((n_pol, ny_full, nx_full))

        for i_pol in range(n_pol):
            ny_c, nx_c = d_zfs_coarse[i_pol].shape
            zoom_y = ny_full / ny_c
            zoom_x = nx_full / nx_c
            zoomed = zoom(
                d_zfs_coarse[i_pol],
                (zoom_y, zoom_x),
                order=self._settings.interpolation_order,
            )
            # Trim to exact target shape (guards against float rounding)
            result[i_pol] = zoomed[:ny_full, :nx_full]

        return result

    def _fold_spectra(
        self,
        d_zfs_map: NDArray,
    ) -> tuple[xr.DataArray, xr.DataArray, NDArray]:
        """Fold all spectra using the per-pixel D_ZFS map.

        Uses a common df axis derived from the median D_ZFS so that all pixels
        share the same frequency coordinate in the output.

        Returns:
            (folded, antisymmetric, delta_f_ghz)
            - folded: xr.DataArray (polarity, y, x, freq_idx) with delta_f_ghz coord
            - antisymmetric: same shape -- S_low - S_high (near zero for good data)
            - delta_f_ghz: 1D NDArray of Zeeman offsets in GHz
        """
        data = self._odmr_data.data
        freq_ghz = data.coords["freq_ghz"].values  # (2, n_freq)
        f_low = freq_ghz[0]
        f_high = freq_ghz[1]
        step = float(np.median(np.diff(f_low)))

        # Common df axis from median D_ZFS
        d_ref = float(np.median(d_zfs_map))
        df_inner, df_outer = _overlap_range(f_low, f_high, d_ref)

        n_pts = int((df_outer - df_inner) / step)
        if n_pts < self._settings.min_overlap_points:
            msg = (
                f"Only {n_pts} overlap points at median D={d_ref:.6f} GHz, "
                f"need >= {self._settings.min_overlap_points}."
            )
            raise FoldingOverlapError(msg)

        delta_f = np.arange(df_inner, df_outer, step)  # (n_df,)
        n_df = len(delta_f)

        n_pol, _, ny, nx, _ = data.shape

        # Spectra arrays: (n_pol, ny, nx, n_freq)
        spectra_low = data.sel(freq_range="low").values  # (n_pol, ny, nx, n_freq)
        spectra_high = data.sel(freq_range="high").values  # (n_pol, ny, nx, n_freq)

        # Per-pixel query frequencies: (n_pol, ny, nx, n_df)
        query_low = d_zfs_map[..., np.newaxis] - delta_f[np.newaxis, np.newaxis, np.newaxis, :]
        query_high = d_zfs_map[..., np.newaxis] + delta_f[np.newaxis, np.newaxis, np.newaxis, :]

        # Flatten batch dim for _interp_batch
        n_total = n_pol * ny * nx
        flat_low = spectra_low.reshape(n_total, -1)  # (N, n_freq)
        flat_high = spectra_high.reshape(n_total, -1)  # (N, n_freq)
        flat_q_low = query_low.reshape(n_total, n_df)  # (N, n_df)
        flat_q_high = query_high.reshape(n_total, n_df)  # (N, n_df)

        s_low_interp = _interp_batch(flat_low, f_low, flat_q_low)  # (N, n_df)
        s_high_interp = _interp_batch(flat_high, f_high, flat_q_high)  # (N, n_df)

        # Reshape back to (n_pol, ny, nx, n_df)
        s_low_4d = s_low_interp.reshape(n_pol, ny, nx, n_df)
        s_high_4d = s_high_interp.reshape(n_pol, ny, nx, n_df)

        folded_arr = (s_low_4d + s_high_4d) / 2.0
        anti_arr = s_low_4d - s_high_4d

        pol_labels = list(data.coords["polarity"].values)

        folded_da = xr.DataArray(
            folded_arr,
            dims=("polarity", "y", "x", "freq_idx"),
            coords={
                "polarity": pol_labels,
                "delta_f_ghz": ("freq_idx", delta_f),
            },
        )
        anti_da = xr.DataArray(
            anti_arr,
            dims=("polarity", "y", "x", "freq_idx"),
            coords={
                "polarity": pol_labels,
                "delta_f_ghz": ("freq_idx", delta_f),
            },
        )

        return folded_da, anti_da, delta_f

    def _compute_fold_residual(
        self,
        antisymmetric: xr.DataArray,
        folded: xr.DataArray,
    ) -> xr.DataArray:
        """Compute normalised fold residual map.

        residual = mean(anti^2, axis=freq) / (var(folded, axis=freq) + eps)
        clipped to [0, 1].

        Returns:
            xr.DataArray of shape (polarity, y, x) in [0, 1].
            Low values indicate good spectral symmetry.
        """
        eps = 1e-12
        anti = antisymmetric.values  # (n_pol, ny, nx, n_df)
        fold = folded.values  # (n_pol, ny, nx, n_df)

        numerator = np.mean(anti**2, axis=-1)  # (n_pol, ny, nx)
        denominator = np.var(fold, axis=-1) + eps  # (n_pol, ny, nx)
        residual = np.clip(numerator / denominator, 0.0, 1.0)

        pol_labels = list(antisymmetric.coords["polarity"].values)
        return xr.DataArray(
            residual,
            dims=("polarity", "y", "x"),
            coords={"polarity": pol_labels},
        )
