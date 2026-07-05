"""Spectral folding for ODMR data (QEP-011).

Exploits the f+/f- mirror symmetry of NV-center ODMR spectra about D_ZFS:

    f+/- = D +/- gamma*B*cos(theta)

The low frequency range captures f- and the high range captures f+. Folding
them together about D produces a per-pixel D_ZFS map (temperature/strain),
a model-free quality metric (fold residual), and halves the number of fit
calls (one folded spectrum per polarity instead of two branches).

**SNR trade-off:** The folded spectrum has sqrt(2) lower noise per frequency
point because it averages two independent measurements. However, the D_ZFS
estimation step introduces per-pixel errors that propagate into the folded
spectrum and subsequent fit. For strong B111 signals (std >> 2-3 uT), this
error is negligible and folding gives a net accuracy benefit. For weak
signals (B111 std < 2 uT), the D_ZFS estimation error can dominate and
the normal (unfolded) fit may be more accurate.

**D_ZFS estimation methods:**

- ``centroid`` (default): Absorption-weighted centroid of each branch.
  O(N) per pixel, no coarsening needed, ~0.06-0.16 MHz RMSE. The centroid
  power is isotope-dependent: power=2 for ESR15N (concentrates weight on
  the doublet), power=1 for ESR14N (avoids catastrophic failure from unequal
  hyperfine peak depths).

- ``brute_force``: Original two-scale algorithm. Coarsens spatially, sweeps
  D candidates, then bicubic-interpolates to full resolution. Slower and
  ~1 MHz RMSE, but preserved for reproducibility with older results.

Algorithm (centroid -- default):
    1. Compute absorption centroid per branch at full resolution
    2. D_ZFS = (centroid_low + centroid_high) / 2
    3. Per-pixel fold: S_folded(df) = (S_low(D-df) + S_high(D+df)) / 2
    4. Fold residual: measure of spectrum asymmetry (quality map)

Algorithm (brute_force):
    1. Spatially coarsen (bin_factor^2) -> high SNR for coarse D_ZFS estimation
    2. Brute-force search over D candidates -> coarse D_ZFS map
    3. Bicubic-interpolate coarse map to full resolution
    4. Per-pixel fold: S_folded(df) = (S_low(D-df) + S_high(D+df)) / 2
    5. Fold residual: measure of spectrum asymmetry (quality map)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import zoom

from qdmpy.constants import D_ZFS
from qdmpy.exceptions import DataValidationError, FoldingOverlapError

if TYPE_CHECKING:
    from qdmpy.odmr.data import ODMRData

_MIN_FREQ_RANGES = 2

# ---------------------------------------------------------------------------
# Centroid power mapping (isotope-dependent)
# ---------------------------------------------------------------------------

_CENTROID_POWER: dict[str, float] = {
    "ESR15N": 2.0,
    "ESR14N": 1.0,
    "ESRSINGLE": 1.0,
}
_DEFAULT_CENTROID_POWER = 1.0


def _resolve_centroid_power(model_name: str | None) -> float:
    """Look up the optimal centroid power for a given ESR model.

    ESR15N uses power=2 (concentrates weight on the doublet peaks).
    ESR14N uses power=1 (power=2 can fail catastrophically when hyperfine
    peak depths are unequal). Unknown models default to power=1 (safe).

    Args:
        model_name: ESR model name, or None for the safe default.

    Returns:
        Centroid power exponent.
    """
    if model_name is None:
        return _DEFAULT_CENTROID_POWER
    return _CENTROID_POWER.get(model_name, _DEFAULT_CENTROID_POWER)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class FoldingSettings(BaseModel):
    """Immutable settings for the spectral folding pipeline.

    The ``d_zfs_method`` controls how the per-pixel D_ZFS fold centre is
    estimated:

    - ``'auto'`` (default): selects the centroid method, which is faster
      and more accurate than brute-force in all tested cases.
    - ``'centroid'``: explicit centroid selection. Runs at full resolution
      in O(N) with ~0.06-0.16 MHz RMSE.
    - ``'brute_force'``: original two-scale algorithm. Preserves exact
      reproducibility with older results. Uses ``search_range``,
      ``search_steps``, ``bin_factor``, and ``interpolation_order``.

    The brute-force-specific fields (``search_range``, ``search_steps``,
    ``bin_factor``, ``interpolation_order``) are only used when
    ``d_zfs_method='brute_force'``.

    Attributes:
        d_zfs_method: D_ZFS estimation method. 'auto' and 'centroid' use the
            absorption centroid; 'brute_force' uses the original search.
        d_zfs_initial: Starting centre for D_ZFS search in GHz.
        search_range: Half-width of brute-force search in GHz (+/-5 MHz = +/-68 K).
        search_steps: Number of candidate D values in the search grid.
        bin_factor: Spatial binning factor for coarse D_ZFS estimation.
            Lower values give per-pixel accuracy but more noise sensitivity.
        interpolation_order: scipy.ndimage.zoom order (3 = bicubic).
        min_overlap_points: Minimum df points required; raises FoldingOverlapError otherwise.
    """

    model_config = ConfigDict(frozen=True)

    d_zfs_method: Literal["auto", "centroid", "brute_force"] = "auto"
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

    The folded spectrum averages two independent measurements (low and high
    branches) for sqrt(2) noise reduction per frequency point. The D_ZFS map
    provides per-pixel zero-field splitting (sensitive to temperature and
    strain). The fold residual is a model-free quality metric: low values
    indicate good spectral symmetry.

    Note:
        D_ZFS estimation error (~0.05-0.15 MHz) propagates into the folded
        spectrum and subsequent fit. This is negligible for strong B111
        signals (std >> 2-3 uT) but can dominate for weak signals, making
        the normal (unfolded) fit more accurate in that regime.

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
    d_zfs_estimation_method: str | None = None

    def to_fit_inputs(self) -> tuple[xr.DataArray, NDArray]:
        """Build the 5D DataArray and absolute-GHz frequency array for fitting.

        Converts the folded spectrum (delta_f domain) into the same format
        expected by FitManager.fit() and refit_outliers(): a 5D xr.DataArray
        with dims (polarity, freq_range, y, x, freq_idx) and a 2D frequency
        array of shape (1, n_freq) in absolute GHz (D_ZFS + delta_f).

        Returns:
            Tuple of (data_xr, frequencies) ready for fitting/refitting.
        """
        spec_vals = self.folded_spectrum.values  # (n_pol, ny, nx, n_df)
        delta_f_ghz: NDArray = self.folded_spectrum.coords["delta_f_ghz"].values
        abs_freq_ghz = D_ZFS + delta_f_ghz
        pol_labels = list(self.folded_spectrum.coords["polarity"].values)

        data_5d = np.expand_dims(spec_vals, axis=1)
        data_xr = xr.DataArray(
            data_5d,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={"polarity": pol_labels, "freq_range": ["folded"]},
        )
        frequencies = abs_freq_ghz.reshape(1, -1)
        return data_xr, frequencies

    def plot(self) -> None:
        """Quick diagnostic overview of the folding result."""
        from qdmpy.plotting import plot_folding_overview

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


def _estimate_d_zfs_centroid(data: xr.DataArray, power: float) -> NDArray:
    """Estimate per-pixel D_ZFS from absorption-weighted centroids.

    Computes the weighted-mean frequency of the absorption dip in each
    branch (low and high), then averages: D = (centroid_low + centroid_high) / 2.
    Fully vectorized, O(N) per pixel, no spatial coarsening needed.

    The ``power`` exponent controls how much weight concentrates on the
    deepest part of the dip vs. the broad tails:

    - power=1: safe for all isotopes, especially ESR14N where unequal
      hyperfine peak depths can bias power=2 toward one peak.
    - power=2: better accuracy for ESR15N (doublet with equal peaks).

    Args:
        data: ODMR data with shape (n_pol, n_frange, y, x, freq_idx)
            and a ``freq_ghz`` coordinate of shape (n_frange, n_freq).
        power: Exponent applied to absorption weights before normalisation.

    Returns:
        NDArray of shape (n_pol, ny, nx) with per-pixel D_ZFS in GHz.
    """
    freq_ghz = data.coords["freq_ghz"].values  # (2, n_freq)
    f_low = freq_ghz[0]
    f_high = freq_ghz[1]

    n_pol = data.sizes["polarity"]
    ny = data.sizes["y"]
    nx = data.sizes["x"]
    result = np.empty((n_pol, ny, nx))

    for i_pol in range(n_pol):
        spec_low = data.isel(polarity=i_pol).sel(freq_range="low").values
        spec_high = data.isel(polarity=i_pol).sel(freq_range="high").values
        # Flatten spatial dims: (ny, nx, n_freq) -> (n_pixel, n_freq)
        flat_low = spec_low.reshape(-1, spec_low.shape[-1])
        flat_high = spec_high.reshape(-1, spec_high.shape[-1])

        # Absorption = max - spectrum, clipped >= 0, raised to power
        abs_low = np.maximum(flat_low.max(axis=1, keepdims=True) - flat_low, 0.0) ** power
        abs_high = np.maximum(flat_high.max(axis=1, keepdims=True) - flat_high, 0.0) ** power

        # Normalise to weights
        w_low = abs_low / (abs_low.sum(axis=1, keepdims=True) + 1e-12)
        w_high = abs_high / (abs_high.sum(axis=1, keepdims=True) + 1e-12)

        # Weighted mean frequency
        centroid_low = (w_low * f_low[np.newaxis, :]).sum(axis=1)
        centroid_high = (w_high * f_high[np.newaxis, :]).sum(axis=1)

        result[i_pol] = ((centroid_low + centroid_high) / 2.0).reshape(ny, nx)

    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SpectralFolder:
    """Folds ODMR spectra about the per-pixel D_ZFS value.

    The folded spectrum has sqrt(2) lower noise per frequency point.
    However, the D_ZFS estimation step introduces errors (~0.05-0.15 MHz
    depending on method and linewidth) that propagate into the fitted B111.
    For strong B111 signals this is negligible; for weak signals (std < 2 uT)
    the normal (unfolded) fit may give better B111 accuracy.

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
        model_name: str | None = None,
    ) -> None:
        """Initialise SpectralFolder.

        Args:
            odmr_data: ODMR data with both 'low' and 'high' frequency ranges.
            settings: Folding configuration. Defaults to FoldingSettings().
            model_name: ESR model name (e.g. 'ESR15N', 'ESR14N') used to
                select the optimal centroid power. None uses the safe default
                (power=1.0).
        """
        self._odmr_data = odmr_data
        self._settings = settings if settings is not None else FoldingSettings()
        self._model_name = model_name

    def fold(self) -> FoldedODMR:
        """Run the folding pipeline and return a FoldedODMR result.

        The D_ZFS estimation method is controlled by
        ``self._settings.d_zfs_method``:

        - ``'auto'`` / ``'centroid'``: absorption centroid at full resolution.
        - ``'brute_force'``: coarsen -> search -> interpolate (original).

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

        method = self._settings.d_zfs_method

        if method in ("auto", "centroid"):
            d_zfs_full = self._estimate_d_zfs_via_centroid()
            d_candidates = None
            search_residual = None
            estimation_method = "centroid"
        else:
            logger.debug(
                "SpectralFolder.fold(): data shape={}, bin_factor={}, search_steps={}",
                data.shape,
                self._settings.bin_factor,
                self._settings.search_steps,
            )
            coarse = self._coarsen_data()
            logger.debug(
                "Coarse shape: {} x {} (from {} x {})",
                coarse.sizes["y"],
                coarse.sizes["x"],
                data.sizes["y"],
                data.sizes["x"],
            )
            d_zfs_coarse, d_candidates, search_residual = self._find_d_zfs_coarse(coarse)
            logger.debug(
                "Coarse D_ZFS: {:.6f} - {:.6f} GHz",
                float(d_zfs_coarse.min()),
                float(d_zfs_coarse.max()),
            )
            target_shape = (data.sizes["y"], data.sizes["x"])
            d_zfs_full = self._interpolate_d_zfs(d_zfs_coarse, target_shape)
            estimation_method = "brute_force"

        # Fold spectra and compute residual (shared by both methods)
        folded, antisymmetric, _delta_f = self._fold_spectra(d_zfs_full)
        fold_residual = self._compute_fold_residual(antisymmetric, folded)

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
            d_zfs_estimation_method=estimation_method,
        )

    # -----------------------------------------------------------------------
    # Private pipeline steps
    # -----------------------------------------------------------------------

    def _estimate_d_zfs_via_centroid(self) -> NDArray:
        """Estimate D_ZFS at full resolution using absorption centroids.

        Returns:
            NDArray of shape (n_pol, ny, nx) with per-pixel D_ZFS in GHz.
        """
        power = _resolve_centroid_power(self._model_name)
        logger.info(
            "Centroid D_ZFS estimation: model={}, power={}",
            self._model_name or "auto",
            power,
        )
        d_zfs = _estimate_d_zfs_centroid(self._odmr_data.data, power=power)
        logger.debug(
            "Centroid D_ZFS: {:.6f} - {:.6f} GHz",
            float(d_zfs.min()),
            float(d_zfs.max()),
        )
        return d_zfs

    def _coarsen_data(self) -> xr.DataArray:
        """Coarsen ODMR data spatially by bin_factor for high-SNR D_ZFS search."""
        bf = self._settings.bin_factor
        return self._odmr_data.data.coarsen(y=bf, x=bf, boundary="trim").mean()  # type: ignore[attr-defined]

    def _find_d_zfs_coarse(self, coarse_data: xr.DataArray) -> tuple[NDArray, NDArray, NDArray]:
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
        logger.info(
            "Brute-force D_ZFS search: {} candidates in [{:.6f}, {:.6f}] GHz",
            settings.search_steps,
            d_candidates[0],
            d_candidates[-1],
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
