"""Tests for qdmpy.odmr.analysis — b111_from_dip_positions."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from qdmpy.constants import GAMMA_NV
from qdmpy.exceptions import DataValidationError
from qdmpy.odmr.analysis import b111_from_dip_positions


def _make_odmr_data(
    dip_low_neg: float,
    dip_high_neg: float,
    dip_low_pos: float,
    dip_high_pos: float,
    n_freq: int = 30,
    h: int = 4,
    w: int = 4,
) -> xr.DataArray:
    """Build a synthetic 5D ODMR DataArray with a single dip per range/polarity.

    The dip is placed at the specified frequency by making that point the
    minimum intensity value.
    """
    freq_low = np.linspace(2.72, 2.87, n_freq)
    freq_high = np.linspace(2.87, 3.02, n_freq)
    freq_ghz = np.stack([freq_low, freq_high])  # (2, n_freq)

    data = np.ones((2, 2, h, w, n_freq), dtype=np.float64)

    dips = {
        (0, 0): dip_low_neg,  # neg, low
        (0, 1): dip_high_neg,  # neg, high
        (1, 0): dip_low_pos,  # pos, low
        (1, 1): dip_high_pos,  # pos, high
    }
    for (ipol, ifr), dip_freq in dips.items():
        idx = int(np.argmin(np.abs(freq_ghz[ifr] - dip_freq)))
        data[ipol, ifr, :, :, idx] = 0.5  # dip

    return xr.DataArray(
        data,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={
            "polarity": ["neg", "pos"],
            "freq_range": ["low", "high"],
            "freq_ghz": (("freq_range", "freq_idx"), freq_ghz),
        },
    )


class TestB111FromDipPositions:
    """Tests for the quick B111 estimator."""

    def test_output_keys(self) -> None:
        da = _make_odmr_data(2.82, 2.92, 2.82, 2.92)
        result = b111_from_dip_positions(da)
        assert "remanent" in result
        assert "induced" in result

    def test_output_shape(self) -> None:
        da = _make_odmr_data(2.82, 2.92, 2.82, 2.92, h=6, w=8)
        result = b111_from_dip_positions(da)
        assert result["remanent"].shape == (6, 8)
        assert result["induced"].shape == (6, 8)

    def test_symmetric_dips_zero_remanent(self) -> None:
        """Identical dips for neg/pos -> induced only, remanent ~ 0."""
        da = _make_odmr_data(2.82, 2.92, 2.82, 2.92)
        result = b111_from_dip_positions(da)
        np.testing.assert_allclose(result["remanent"], 0.0, atol=0.5)

    def test_missing_polarity_raises(self) -> None:
        da = _make_odmr_data(2.82, 2.92, 2.82, 2.92)
        da_neg_only = da.sel(polarity="neg")
        da_1d = da_neg_only.expand_dims("polarity")
        with pytest.raises(DataValidationError, match="polarity"):
            b111_from_dip_positions(da_1d)

    def test_missing_freq_range_raises(self) -> None:
        da = _make_odmr_data(2.82, 2.92, 2.82, 2.92)
        da_low = da.sel(freq_range="low")
        da_1d = da_low.expand_dims("freq_range")
        with pytest.raises(DataValidationError, match="freq_range"):
            b111_from_dip_positions(da_1d)

    def test_known_splitting_gives_expected_field(self) -> None:
        """A known 0.10 GHz splitting should give B = 0.1 / 2 / GAMMA_NV * 1e6 uT."""
        # Put dips at 2.82 (low) and 2.92 (high) -> splitting = 0.10 GHz
        da = _make_odmr_data(2.82, 2.92, 2.82, 2.92)
        result = b111_from_dip_positions(da)
        freq_low = np.linspace(2.72, 2.87, 30)
        freq_high = np.linspace(2.87, 3.02, 30)
        # actual dip positions are the nearest grid points
        actual_low = freq_low[int(np.argmin(np.abs(freq_low - 2.82)))]
        actual_high = freq_high[int(np.argmin(np.abs(freq_high - 2.92)))]
        expected_db = (actual_high - actual_low) / 2.0 / GAMMA_NV * 1e6
        # Induced should be close to +/- expected_db (symmetric case)
        np.testing.assert_allclose(
            np.abs(result["induced"]),
            expected_db,
            atol=1.0,  # grid quantization tolerance
        )
