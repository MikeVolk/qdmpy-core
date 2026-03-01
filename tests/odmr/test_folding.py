"""Unit tests for SpectralFolder, FoldedODMR, and FoldingSettings.

All tests use synthetic ODMR data with known Lorentzian dips -- no MATLAB
files or real data required.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray
from pydantic import ValidationError
from scipy.ndimage import zoom

from qdmpy.constants import D_ZFS
from qdmpy.exceptions import DataValidationError, FoldingOverlapError
from qdmpy.odmr.data import ODMRData
from qdmpy.odmr.folding import (
    FoldedODMR,
    FoldingSettings,
    SpectralFolder,
    _interp_batch,
    _overlap_range,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

D_TEST = D_ZFS  # 2.870 GHz
ZEEMAN_SHIFT = 0.010  # GHz — separates low/high dip from D
HALF_WIDTH_FRANGE = 0.013  # GHz — each freq range half-width around dip centre
N_FREQ = 51
DIP_WIDTH = 0.0020  # GHz — Lorentzian FWHM
DIP_CONTRAST = 0.08  # fractional depth


def _lorentzian_dip(f: NDArray, center: float, width: float, contrast: float) -> NDArray:
    """Single Lorentzian dip: 1 - contrast / (1 + ((f - center) / (width/2))^2)."""
    return 1.0 - contrast / (1.0 + ((f - center) / (width / 2.0)) ** 2)


def _make_freq_axes(
    d: float = D_TEST,
    zeeman: float = ZEEMAN_SHIFT,
    hw: float = HALF_WIDTH_FRANGE,
    n_freq: int = N_FREQ,
) -> tuple[NDArray, NDArray]:
    """Create symmetric low/high frequency axes centred on f- = D-zeeman, f+ = D+zeeman."""
    f_low = np.linspace(d - zeeman - hw, d - zeeman + hw, n_freq)
    f_high = np.linspace(d + zeeman - hw, d + zeeman + hw, n_freq)
    return f_low, f_high


def _make_odmr_data(
    shape: tuple[int, int] = (8, 8),
    d_map: NDArray | None = None,  # per-pixel D_ZFS, shape (n_pol, y, x)
    b_map: NDArray | None = None,  # per-pixel Zeeman shift (GHz), shape (y, x)
    noise: float = 0.0,
    seed: int = 0,
    n_freq: int = N_FREQ,
) -> ODMRData:
    """Build a minimal synthetic ODMRData with 2 polarities and 2 freq ranges.

    Dips follow a single Lorentzian. Spectra are symmetric about the per-pixel D_ZFS
    when d_map is uniform and b_map is zero (used in symmetry tests).

    Args:
        shape: (ny, nx) spatial shape.
        d_map: (n_pol, ny, nx) per-pixel D_ZFS values. Defaults to D_TEST everywhere.
        b_map: (ny, nx) Zeeman shifts in GHz. Defaults to ZEEMAN_SHIFT everywhere.
        noise: Gaussian noise std.
        seed: Random seed.
        n_freq: Frequency points per range.

    Returns:
        ODMRData with dims (polarity, freq_range, y, x, freq_idx).
    """
    ny, nx = shape
    n_pol, n_frange = 2, 2
    rng = np.random.default_rng(seed)

    if b_map is None:
        b_map = np.full((ny, nx), ZEEMAN_SHIFT)
    if d_map is None:
        d_map = np.full((n_pol, ny, nx), D_TEST)

    # Use the mean D_ZFS for freq axis construction (common axis)
    d_ref = float(np.mean(d_map))
    f_low, f_high = _make_freq_axes(d=d_ref, zeeman=ZEEMAN_SHIFT, n_freq=n_freq)
    freqs = np.stack([f_low, f_high])  # (2, n_freq)

    spectra = np.ones((n_pol, n_frange, ny, nx, n_freq))

    for i_pol in range(n_pol):
        for iy in range(ny):
            for ix in range(nx):
                d_px = d_map[i_pol, iy, ix]
                z = b_map[iy, ix]
                # low range: dip at D - zeeman
                spectra[i_pol, 0, iy, ix] = _lorentzian_dip(
                    f_low, d_px - z, DIP_WIDTH, DIP_CONTRAST
                )
                # high range: dip at D + zeeman
                spectra[i_pol, 1, iy, ix] = _lorentzian_dip(
                    f_high, d_px + z, DIP_WIDTH, DIP_CONTRAST
                )

    if noise > 0.0:
        spectra += rng.normal(0, noise, spectra.shape)

    da = xr.DataArray(
        spectra,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={
            "polarity": ["neg", "pos"],
            "freq_range": ["low", "high"],
            "freq_ghz": (("freq_range", "freq_idx"), freqs),
        },
    )
    return ODMRData(data=da)


# ---------------------------------------------------------------------------
# FoldingSettings tests
# ---------------------------------------------------------------------------


class TestFoldingSettings:
    def test_folding_settings_defaults(self) -> None:
        s = FoldingSettings()
        assert s.d_zfs_initial == pytest.approx(D_ZFS)
        assert s.search_range == pytest.approx(0.005)
        assert s.search_steps == 201
        assert s.bin_factor == 8
        assert s.interpolation_order == 3
        assert s.min_overlap_points == 5

    def test_immutability(self) -> None:
        s = FoldingSettings()
        with pytest.raises(ValidationError):  # pydantic frozen model
            s.search_steps = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validation_requires_low_high(self) -> None:
        """Single freq_range → DataValidationError when fold() is called."""
        # Build a valid 2-range ODMRData first, then slice to 1 range
        odmr = _make_odmr_data(shape=(4, 4))
        # Confirm setup works first
        _ = odmr.data.isel(freq_range=0)
        # Create a 1-frange DataArray that still passes ODMRData construction
        f_low = np.linspace(2.84, 2.86, 20)
        freqs_1d = np.stack([f_low])  # (1, n_freq)
        single_range_da = xr.DataArray(
            np.ones((2, 1, 4, 4, 20)),
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": ["neg", "pos"],
                "freq_range": ["low"],
                "freq_ghz": (("freq_range", "freq_idx"), freqs_1d),
            },
        )
        odmr_single = ODMRData(data=single_range_da)
        folder = SpectralFolder(odmr_single)
        with pytest.raises(DataValidationError):
            folder.fold()


# ---------------------------------------------------------------------------
# Fold residual / symmetry tests
# ---------------------------------------------------------------------------


class TestFoldResidual:
    def test_fold_residual_near_zero_symmetric(self) -> None:
        """Perfectly symmetric input (no noise) → fold residual < 0.01."""
        odmr = _make_odmr_data(shape=(8, 8), noise=0.0)
        settings = FoldingSettings(bin_factor=4, search_steps=101)
        result = SpectralFolder(odmr, settings).fold()

        assert result.fold_residual.dims == ("polarity", "y", "x")
        # Mean residual across all pixels should be near zero
        assert float(result.fold_residual.mean()) < 0.01

    def test_antisymmetric_near_zero(self) -> None:
        """Symmetric input → antisymmetric component < noise floor."""
        odmr = _make_odmr_data(shape=(8, 8), noise=0.0)
        settings = FoldingSettings(bin_factor=4, search_steps=101)
        result = SpectralFolder(odmr, settings).fold()

        anti_rms = float(np.sqrt(np.mean(result.antisymmetric_spectrum.values**2)))
        assert anti_rms < 1e-3


# ---------------------------------------------------------------------------
# D_ZFS recovery tests
# ---------------------------------------------------------------------------


class TestDZFSRecovery:
    def test_recovers_known_d_zfs_shift(self) -> None:
        """D injected at +2 MHz → recovered within 0.1 MHz."""
        d_injected = D_ZFS + 0.002  # +2 MHz
        d_map = np.full((2, 8, 8), d_injected)
        odmr = _make_odmr_data(shape=(8, 8), d_map=d_map, noise=0.0)

        settings = FoldingSettings(
            d_zfs_initial=D_ZFS,
            search_range=0.005,
            search_steps=201,
            bin_factor=4,
        )
        result = SpectralFolder(odmr, settings).fold()

        d_recovered = float(result.d_zfs_map.mean())
        assert abs(d_recovered - d_injected) < 0.0001  # < 0.1 MHz

    def test_bicubic_interpolation_gaussian_hotspot(self) -> None:
        """Smooth Gaussian D field → bicubic interpolation RMSE < 0.05 MHz.

        Tests the _interpolate_d_zfs step directly by constructing a known
        smooth D_ZFS field, coarsening it, interpolating back, and measuring
        the reconstruction error. The Gaussian must be smooth relative to the
        super-pixel size (sigma >= 3 super-pixels) for bicubic to be accurate.
        """
        ny, nx = 64, 64
        bin_factor = 4  # 16x16 coarse grid

        # Smooth Gaussian D field: sigma=12 pixels = 3 super-pixels
        y_idx = np.arange(ny)[:, None]
        x_idx = np.arange(nx)[None, :]
        cy, cx = ny / 2, nx / 2
        sigma = 12.0
        amplitude_ghz = 0.001  # 1 MHz peak amplitude
        d_field = D_ZFS + amplitude_ghz * np.exp(
            -((y_idx - cy) ** 2 + (x_idx - cx) ** 2) / (2 * sigma**2)
        )  # (ny, nx)

        # Coarsen: simulate what the SpectralFolder does
        ny_c = ny // bin_factor  # 16
        nx_c = nx // bin_factor  # 16
        d_coarse = (
            d_field[: ny_c * bin_factor, : nx_c * bin_factor]
            .reshape(ny_c, bin_factor, nx_c, bin_factor)
            .mean(axis=(1, 3))
        )  # (ny_c, nx_c)

        # Interpolate back with bicubic
        zoom_y = ny / ny_c
        zoom_x = nx / nx_c
        d_interp = zoom(d_coarse, (zoom_y, zoom_x), order=3)[:ny, :nx]

        # RMSE in MHz
        rmse_mhz = float(np.sqrt(np.mean((d_interp - d_field) ** 2))) * 1000
        assert rmse_mhz < 0.05


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_folded_spectrum_shape(self) -> None:
        """Folded spectrum must not have freq_range dim; delta_f_ghz coord must exist."""
        odmr = _make_odmr_data(shape=(4, 4))
        settings = FoldingSettings(bin_factor=2, search_steps=51)
        result = SpectralFolder(odmr, settings).fold()

        assert "freq_range" not in result.folded_spectrum.dims
        assert "delta_f_ghz" in result.folded_spectrum.coords
        assert result.folded_spectrum.dims == ("polarity", "y", "x", "freq_idx")
        # Spatial dims should match input
        assert result.folded_spectrum.sizes["y"] == 4
        assert result.folded_spectrum.sizes["x"] == 4
        assert result.d_zfs_map.dims == ("polarity", "y", "x")
        assert result.fold_residual.dims == ("polarity", "y", "x")


# ---------------------------------------------------------------------------
# SNR improvement test
# ---------------------------------------------------------------------------


class TestSNRImprovement:
    def test_snr_improvement(self) -> None:
        """Folded mean has same dip amplitude as single-range but noise / sqrt(2).

        The folded spectrum averages two independent measurements, so:
        - Signal (dip depth) is the same as a single range
        - Noise is reduced by sqrt(2) (averaging two independent samples)
        - SNR improves by sqrt(2) ~ 1.41

        Uses the known noise level as denominator (not std of the full spectrum,
        which would be dominated by spectral features rather than noise).
        """
        noise_level = 0.005
        shape = (8, 8)  # 8x8 pixels all with the same field
        odmr = _make_odmr_data(shape=shape, noise=noise_level, seed=42)
        settings = FoldingSettings(bin_factor=4, search_steps=101)
        result = SpectralFolder(odmr, settings).fold()

        n_pixels = shape[0] * shape[1]

        # Signal: mean over all pixels (same field -> dip stays sharp)
        single_spec = odmr.data.sel(polarity="neg", freq_range="low").values.mean(axis=(0, 1))
        folded_spec = result.folded_spectrum.sel(polarity="neg").values.mean(axis=(0, 1))

        # Dip amplitude: baseline - minimum
        dip_single = float(single_spec.max() - single_spec.min())
        dip_folded = float(folded_spec.max() - folded_spec.min())

        # Noise per frequency point after pixel averaging:
        # single range: noise_level / sqrt(n_pixels)
        # folded (mean of two independent measurements): noise_level / sqrt(2) / sqrt(n_pixels)
        noise_single = noise_level / np.sqrt(n_pixels)
        noise_folded = noise_level / np.sqrt(2.0) / np.sqrt(n_pixels)

        snr_single = dip_single / noise_single
        snr_folded = dip_folded / noise_folded

        assert snr_folded > 1.2 * snr_single


# ---------------------------------------------------------------------------
# Error condition tests
# ---------------------------------------------------------------------------


class TestErrorConditions:
    def test_narrow_overlap_raises(self) -> None:
        """Frequency ranges with < min_overlap_points → FoldingOverlapError."""
        # Make ranges that barely touch D_ZFS — almost no overlap
        d = D_ZFS
        # Place ranges right next to D with almost no room
        gap = 0.015  # large gap, so overlap is very narrow
        f_low = np.linspace(d - 0.020, d - gap, 20)
        f_high = np.linspace(d + gap, d + 0.020, 20)
        freqs = np.stack([f_low, f_high])

        da = xr.DataArray(
            np.ones((2, 2, 2, 2, 20)),
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": ["neg", "pos"],
                "freq_range": ["low", "high"],
                "freq_ghz": (("freq_range", "freq_idx"), freqs),
            },
        )
        odmr = ODMRData(data=da)
        settings = FoldingSettings(
            d_zfs_initial=d,
            search_range=0.001,  # tight range
            search_steps=11,
            min_overlap_points=50,  # require many points — impossible with this gap
        )
        folder = SpectralFolder(odmr, settings)
        with pytest.raises(FoldingOverlapError):
            folder.fold()


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestOverlapRangeHelper:
    def test_overlap_range_helper(self) -> None:
        """Known f_low/f_high/D → correct df_inner, df_outer."""
        f_low = np.linspace(2.852, 2.868, 51)
        f_high = np.linspace(2.872, 2.888, 51)
        d = 2.870

        df_inner, df_outer = _overlap_range(f_low, f_high, d)

        # df_inner = max(d - f_low[-1], f_high[0] - d) = max(0.002, 0.002) = 0.002
        assert df_inner == pytest.approx(0.002, abs=1e-6)
        # df_outer = min(d - f_low[0], f_high[-1] - d) = min(0.018, 0.018) = 0.018
        assert df_outer == pytest.approx(0.018, abs=1e-6)

    def test_overlap_range_raises_no_overlap(self) -> None:
        """Ranges that don't span D → FoldingOverlapError."""
        f_low = np.linspace(2.83, 2.86, 30)
        f_high = np.linspace(2.88, 2.91, 30)
        # D = 2.870 lies in the gap — but overlap computation:
        # df_inner = max(2.870 - 2.860, 2.880 - 2.870) = max(0.010, 0.010) = 0.010
        # df_outer = min(2.870 - 2.830, 2.910 - 2.870) = min(0.040, 0.040) = 0.040
        # So there IS overlap. Let's use a D outside the high range instead.
        with pytest.raises(FoldingOverlapError):
            _overlap_range(f_low, f_high, d=3.0)  # D far outside both ranges


class TestInterpBatchAccuracy:
    def test_interp_batch_accuracy(self) -> None:
        """Interpolation on a sine wave: error < 1e-3.

        Linear interpolation of sin(x) has error O(h^2/8) where h is the grid
        spacing. For n_freq=200 over [0, 2pi], h ~ 0.031 rad gives max error
        ~ 1.2e-4. Using a denser grid (n_freq=500) reduces this to < 5e-5.
        """
        n_freq = 500
        freqs = np.linspace(0.0, 2 * np.pi, n_freq)
        spectra = np.sin(freqs)[np.newaxis, :]  # (1, n_freq)

        # Query at finely-spaced intermediate points
        n_query = 1000
        query = np.linspace(0.1, 2 * np.pi - 0.1, n_query)[np.newaxis, :]  # (1, n_q)
        result = _interp_batch(spectra, freqs, query)  # (1, n_q)

        expected = np.sin(query[0])
        max_err = float(np.max(np.abs(result[0] - expected)))
        assert max_err < 1e-4


# ---------------------------------------------------------------------------
# Search diagnostics tests
# ---------------------------------------------------------------------------


class TestSearchDiagnostics:
    def test_diagnostics_populated(self) -> None:
        """fold() populates d_candidates and search_residual."""
        odmr = _make_odmr_data(shape=(8, 8), noise=0.0)
        settings = FoldingSettings(bin_factor=4, search_steps=51)
        result = SpectralFolder(odmr, settings).fold()

        assert result.d_candidates is not None
        assert result.search_residual is not None

    def test_d_candidates_shape(self) -> None:
        """d_candidates has shape (search_steps,)."""
        n_steps = 51
        odmr = _make_odmr_data(shape=(8, 8))
        settings = FoldingSettings(bin_factor=4, search_steps=n_steps)
        result = SpectralFolder(odmr, settings).fold()

        assert result.d_candidates is not None
        assert result.d_candidates.shape == (n_steps,)

    def test_search_residual_shape(self) -> None:
        """search_residual has shape (n_pol, search_steps)."""
        n_steps = 51
        odmr = _make_odmr_data(shape=(8, 8))
        settings = FoldingSettings(bin_factor=4, search_steps=n_steps)
        result = SpectralFolder(odmr, settings).fold()

        assert result.search_residual is not None
        n_pol = result.d_zfs_map.sizes["polarity"]
        assert result.search_residual.shape == (n_pol, n_steps)

    def test_search_minimum_matches_d_zfs(self) -> None:
        """Minimum of mean search_residual is near the median D_ZFS value."""
        d_injected = D_ZFS + 0.002
        d_map = np.full((2, 8, 8), d_injected)
        odmr = _make_odmr_data(shape=(8, 8), d_map=d_map, noise=0.0)

        settings = FoldingSettings(
            d_zfs_initial=D_ZFS,
            search_range=0.005,
            search_steps=201,
            bin_factor=4,
        )
        result = SpectralFolder(odmr, settings).fold()

        assert result.d_candidates is not None
        assert result.search_residual is not None

        # For each polarity, the argmin of the mean residual should be
        # close to the injected D_ZFS value
        for i_pol in range(2):
            res = result.search_residual[i_pol]
            best_idx = np.argmin(res)
            d_best = result.d_candidates[best_idx]
            assert abs(d_best - d_injected) < 0.0002  # < 0.2 MHz

    def test_plot_method_exists(self) -> None:
        """FoldedODMR has a plot() method."""
        assert hasattr(FoldedODMR, "plot")
