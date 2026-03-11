"""Tests for QEP-044 convenience methods.

Covers:
- ODMR.spectrum() — single-pixel spectrum extraction
- ODMR.plot_spectra() — 2×2 multi-panel plot (mocked plt.show)
- FitResult.plot() / FitResult.show() — thin wrappers over plotting module
- QDMResult.plot() / QDMResult.show() — delegation to FitResult
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from qdmpy.fitting.result import FitResult
from qdmpy.odmr.data import ODMRData
from qdmpy.odmr.manager import ODMR
from qdmpy.result import QDMResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_POL, N_FRANGE, H, W, N_FREQ = 2, 2, 10, 10, 20


@pytest.fixture
def odmr() -> ODMR:
    rng = np.random.default_rng(0)
    arr = rng.random((N_POL, N_FRANGE, H, W, N_FREQ))
    freq_ghz = np.linspace(2.82, 2.92, N_FREQ)
    da = xr.DataArray(
        arr,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={
            "polarity": ["neg", "pos"],
            "freq_range": ["low", "high"],
            "freq_ghz": (["freq_range", "freq_idx"], np.stack([freq_ghz, freq_ghz + 0.1])),
        },
    )
    odmr_instance = ODMR(ODMRData(data=da))
    odmr_instance.process_data()
    return odmr_instance


@pytest.fixture
def fit_result() -> FitResult:
    rng = np.random.default_rng(1)
    n_pixels = H * W
    return FitResult(
        parameters={
            "center": rng.uniform(2.82, 2.92, n_pixels),
            "chi2": rng.random(n_pixels),
        },
        scan_dimensions=(H, W),
        pixel_spacing=4e-6,
        model_name="ESRSINGLE",
    )


@pytest.fixture
def qdm_result(fit_result: FitResult) -> QDMResult:
    return QDMResult(fit_result=fit_result)


# ---------------------------------------------------------------------------
# ODMR.spectrum()
# ---------------------------------------------------------------------------


class TestODMRSpectrum:
    def test_returns_tuple_of_two_arrays(self, odmr: ODMR) -> None:
        result = odmr.spectrum(0, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_freq_shape(self, odmr: ODMR) -> None:
        freq, _ = odmr.spectrum(0, 0)
        assert freq.shape == (N_FREQ,)

    def test_intensity_shape(self, odmr: ODMR) -> None:
        _, spec = odmr.spectrum(0, 0)
        assert spec.shape == (N_FREQ,)

    def test_default_polarity_is_neg(self, odmr: ODMR) -> None:
        freq, spec = odmr.spectrum(3, 5)
        freq_neg, spec_neg = odmr.spectrum(3, 5, polarity="neg")
        np.testing.assert_array_equal(freq, freq_neg)
        np.testing.assert_array_equal(spec, spec_neg)

    def test_default_freq_range_is_low(self, odmr: ODMR) -> None:
        freq, spec = odmr.spectrum(3, 5)
        freq_low, spec_low = odmr.spectrum(3, 5, freq_range="low")
        np.testing.assert_array_equal(freq, freq_low)
        np.testing.assert_array_equal(spec, spec_low)

    def test_high_freq_range_uses_different_frequencies(self, odmr: ODMR) -> None:
        freq_low, _ = odmr.spectrum(0, 0, freq_range="low")
        freq_high, _ = odmr.spectrum(0, 0, freq_range="high")
        # High range is offset by 0.1 GHz in our fixture
        assert not np.allclose(freq_low, freq_high)

    def test_pos_polarity_differs_from_neg(self, odmr: ODMR) -> None:
        _, spec_neg = odmr.spectrum(0, 0, polarity="neg")
        _, spec_pos = odmr.spectrum(0, 0, polarity="pos")
        # Random data — different polarity → different spectrum
        assert not np.allclose(spec_neg, spec_pos)

    def test_processed_false_uses_raw_data(self, odmr: ODMR) -> None:
        freq_raw, spec_raw = odmr.spectrum(0, 0, processed=False)
        assert freq_raw.shape == (N_FREQ,)
        assert spec_raw.shape == (N_FREQ,)

    def test_different_pixels_differ(self, odmr: ODMR) -> None:
        _, spec_a = odmr.spectrum(0, 0)
        _, spec_b = odmr.spectrum(5, 7)
        assert not np.allclose(spec_a, spec_b)


# ---------------------------------------------------------------------------
# ODMR.plot_spectra()
# ---------------------------------------------------------------------------


class TestODMRPlotSpectra:
    def test_calls_plt_show(self, odmr: ODMR) -> None:
        with patch("matplotlib.pyplot.show") as mock_show:
            odmr.plot_spectra(0, 0)
            mock_show.assert_called_once()

    def test_creates_correct_subplot_grid(self, odmr: ODMR) -> None:
        with (
            patch("matplotlib.pyplot.show"),
            patch(
                "matplotlib.pyplot.subplots",
                wraps=__import__("matplotlib.pyplot", fromlist=["subplots"]).subplots,
            ) as mock_subplots,
        ):
            odmr.plot_spectra(0, 0)
            call_kwargs = mock_subplots.call_args
            # 2 polarities × 2 freq_ranges
            assert call_kwargs[0][0] == N_POL
            assert call_kwargs[0][1] == N_FRANGE


# ---------------------------------------------------------------------------
# FitResult.plot() and FitResult.show()
# ---------------------------------------------------------------------------


class TestFitResultPlot:
    def test_plot_calls_plot_fit_result_parameter_map(self, fit_result: FitResult) -> None:
        with patch("qdmpy.plotting.plot_fit_result_parameter_map") as mock:
            fit_result.plot("chi2")
            mock.assert_called_once_with(fit_result, "chi2", save=False, filename=None)

    def test_plot_default_param_is_center(self, fit_result: FitResult) -> None:
        with patch("qdmpy.plotting.plot_fit_result_parameter_map") as mock:
            fit_result.plot()
            mock.assert_called_once_with(fit_result, "center", save=False, filename=None)

    def test_plot_forwards_kwargs(self, fit_result: FitResult) -> None:
        with patch("qdmpy.plotting.plot_fit_result_parameter_map") as mock:
            fit_result.plot("chi2", save=True, filename="out.png")
            mock.assert_called_once_with(fit_result, "chi2", save=True, filename="out.png")

    def test_show_calls_plot_fit_result_overview(self, fit_result: FitResult) -> None:
        with patch("qdmpy.plotting.plot_fit_result_overview") as mock:
            fit_result.show()
            mock.assert_called_once_with(fit_result, save=False, filename=None)

    def test_show_forwards_kwargs(self, fit_result: FitResult) -> None:
        with patch("qdmpy.plotting.plot_fit_result_overview") as mock:
            fit_result.show(save=True)
            mock.assert_called_once_with(fit_result, save=True, filename=None)


# ---------------------------------------------------------------------------
# QDMResult.plot() and QDMResult.show() were removed in QEP-008.
# See tests/test_io_qdm.py::TestPublicApi::test_qdm_result_has_no_io_methods
# ---------------------------------------------------------------------------
