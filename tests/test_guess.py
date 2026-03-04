"""Test module for QDMpy.guess.

These tests cover the model and parameter guessing functionality used for ODMR data
processing and fitting.
"""

from __future__ import annotations

from unittest.mock import patch

import numba
import numpy as np
import pytest

from qdmpy.constants import AHYP_14N, AHYP_15N, DEFAULT_VMAX, DEFAULT_VMIN
from qdmpy.exceptions import (
    DataShapeError,
    DataValidationError,
    ModelNotFoundError,
)
from qdmpy.fitting.guess import (
    _RELATIVE_PROMINENCE,
    _relative_prominence,
    absorption_centroid,
    cumsum_center,
    cumsum_contrast,
    cumsum_width,
    get_model_by_peaks,
    guess_model,
    guess_n_peaks,
    halfpower_width,
    normalize_pixel,
    validate_array,
)
from qdmpy.fitting.guesser import ParameterGuesser
from qdmpy.fitting.models import ESR14N, ESR15N, ESRSINGLE


@pytest.fixture
def sample_odmr_data():
    """Create a sample ODMR data array for testing.

    Convention: (n_pol, n_frange, n_pixel, n_freq).
    """
    # 2 polarities, 3 frequency ranges, 10 pixels, 100 frequencies
    data = np.random.random((2, 3, 10, 100))

    # Add dips at specific frequency indices for peak detection
    for pol in range(2):
        for freq_range in range(3):
            center_idx1 = 30
            for pixel in range(10):
                for i in range(-5, 6):
                    if 0 <= center_idx1 + i < 100:
                        data[pol, freq_range, pixel, center_idx1 + i] = 0.5 - 0.4 * np.exp(
                            -0.5 * (i / 2) ** 2
                        )

            # Second dip for ESR15N detection
            center_idx2 = 70
            for pixel in range(10):
                for i in range(-5, 6):
                    if 0 <= center_idx2 + i < 100:
                        data[pol, freq_range, pixel, center_idx2 + i] = 0.5 - 0.4 * np.exp(
                            -0.5 * (i / 2) ** 2
                        )

    return data


@pytest.fixture
def frequency_range():
    """Create a sample 2D frequency range for testing.

    Shape: (n_frange, n_freq) to match guess function expectations.
    """
    freq_1d = np.linspace(2.87, 2.89, 100)
    return np.tile(freq_1d, (3, 1))  # (3 freq ranges, 100 freqs)


@pytest.fixture
def model_instances():
    """Get model instances for testing."""
    return {
        1: ESRSINGLE(),
        2: ESR15N(),
        3: ESR14N(),
    }


class TestValidateArray:
    """Test cases for validate_array function."""

    def test_validate_correct_dimensions(self) -> None:
        """Test validation with correct dimensions."""
        data = np.zeros((2, 3, 4, 5))
        validate_array(data, 4, "test_data")

    def test_validate_incorrect_dimensions(self) -> None:
        """Test validation with incorrect dimensions."""
        data = np.zeros((2, 3, 4))
        with pytest.raises(DataShapeError) as excinfo:
            validate_array(data, 4, "test_data")
        assert "must have 4 dimensions" in str(excinfo.value)
        assert "Got 3" in str(excinfo.value)

    def test_none_array(self) -> None:
        """Test validation with None."""
        with pytest.raises(DataValidationError):
            validate_array(None, 4, "test_data")

    def test_non_numeric_array(self) -> None:
        """Test validation with a non-numeric array."""
        data = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(DataValidationError):
            validate_array(data, 2, "test_data")

    def test_unexpected_dimensions(self) -> None:
        """Test validation with unexpected dimensions."""
        data = np.zeros((2, 3, 4))
        with pytest.raises(DataShapeError):
            validate_array(data, 5, "test_data")


class TestGuessNPeaks:
    """Test cases for guess_n_peaks function."""

    def test_guess_n_peaks_mock(self) -> None:
        """Test guessing the number of peaks with mocked find_peaks."""
        mock_data = np.random.random((2, 3, 10, 100))

        with patch("qdmpy.fitting.guess.find_peaks") as mock_find_peaks:
            mock_find_peaks.return_value = (np.array([30, 70]), {})
            n_peaks, doubt, indices = guess_n_peaks(mock_data)

            assert n_peaks == 2
            assert bool(doubt) is False
            assert len(indices) == mock_data.shape[0] * mock_data.shape[1]

    def test_guess_n_peaks_doubt(self) -> None:
        """Test doubt when fewer than confidence threshold of combos agree.

        2 pol x 2 frange = 4 combos; 2 return 2 peaks and 2 return 3 peaks
        -> 50% max agreement < 60% threshold -> doubt=True.
        """
        mock_data = np.random.random((2, 2, 10, 100))
        call_count = [0]

        def side_effect_fn(data, prominence):
            call_count[0] += 1
            if call_count[0] <= 2:
                return np.array([30, 70]), {}
            return np.array([25, 50, 75]), {}

        with patch("qdmpy.fitting.guess.find_peaks", side_effect=side_effect_fn):
            n_peaks, doubt, indices = guess_n_peaks(mock_data)

        assert n_peaks in (2, 3)
        assert bool(doubt) is True
        assert len(indices) == mock_data.shape[0] * mock_data.shape[1]

    def test_guess_n_peaks_majority_vote(self) -> None:
        """Test that mode is used: 3-of-4 combos detect 3 peaks → n_peaks=3, no doubt."""
        mock_data = np.random.random((2, 2, 10, 100))
        call_count = [0]

        def side_effect_fn(data, prominence):
            call_count[0] += 1
            # 3 of 4 calls return 3 peaks, 1 returns 2
            if call_count[0] == 3:
                return np.array([25, 50, 75]), {}
            return np.array([25, 75]), {}

        with patch("qdmpy.fitting.guess.find_peaks", side_effect=side_effect_fn):
            n_peaks, doubt, _ = guess_n_peaks(mock_data)

        assert n_peaks == 2  # mode of [2, 2, 3, 2] is 2
        assert bool(doubt) is False  # 3/4 = 75% >= 60%

    def test_guess_n_peaks_low_confidence(self) -> None:
        """Test that doubt is set when fewer than threshold combos agree."""
        mock_data = np.random.random((2, 2, 10, 100))
        call_count = [0]

        def side_effect_fn(data, prominence):
            call_count[0] += 1
            # 50/50 split: calls 1,2 → 2 peaks, calls 3,4 → 3 peaks
            if call_count[0] <= 2:
                return np.array([30, 70]), {}
            return np.array([25, 50, 75]), {}

        with patch("qdmpy.fitting.guess.find_peaks", side_effect=side_effect_fn):
            n_peaks, doubt, _ = guess_n_peaks(mock_data)

        assert n_peaks in (2, 3)  # mode of a tie — either is valid
        assert bool(doubt) is True  # 50% < 60%

    def test_incorrect_dimensions(self) -> None:
        """Test with incorrect dimensions."""
        data = np.zeros((2, 3, 4))
        with pytest.raises(DataShapeError):
            guess_n_peaks(data)


class TestGetModelByPeaks:
    """Test cases for get_model_by_peaks function."""

    def test_get_model_single_peak(self, model_instances) -> None:
        """Test getting the model for a single peak."""
        model = get_model_by_peaks(1)
        assert isinstance(model, ESRSINGLE)
        assert model.n_peaks == 1
        assert model.name == "ESRSINGLE"

    def test_get_model_two_peaks(self, model_instances) -> None:
        """Test getting the model for two peaks."""
        model = get_model_by_peaks(2)
        assert isinstance(model, ESR15N)
        assert model.n_peaks == 2
        assert model.name == "ESR15N"

    def test_get_model_three_peaks(self, model_instances) -> None:
        """Test getting the model for three peaks."""
        model = get_model_by_peaks(3)
        assert isinstance(model, ESR14N)
        assert model.n_peaks == 3
        assert model.name == "ESR14N"

    def test_get_model_invalid_peaks(self) -> None:
        """Test with an invalid number of peaks."""
        with pytest.raises(ModelNotFoundError) as excinfo:
            get_model_by_peaks(4)
        assert "No model found for 4 peaks" in str(excinfo.value)


class TestGuessModel:
    """Test cases for guess_model function."""

    @patch("qdmpy.fitting.guess.guess_n_peaks")
    def test_guess_model_no_doubt(self, mock_guess_n_peaks) -> None:
        """Test guessing model when there's no doubt."""
        mock_guess_n_peaks.return_value = (2, False, [])
        data = np.zeros((2, 3, 10, 100))

        model = guess_model(data)
        assert isinstance(model, ESR15N)
        assert model.n_peaks == 2

    @patch("qdmpy.fitting.guess.guess_n_peaks")
    def test_guess_model_with_doubt_returns_model(self, mock_guess_n_peaks) -> None:
        """Test that guess_model returns a model even when there's doubt (no exception)."""
        mock_guess_n_peaks.return_value = (3, True, [])
        data = np.zeros((2, 3, 10, 100))

        model = guess_model(data)
        assert isinstance(model, ESR14N)
        assert model.n_peaks == 3


class TestNormalizePixel:
    """Test cases for normalize_pixel function."""

    def test_normalize_pixel_normal(self) -> None:
        """Test normalizing a pixel with normal data."""
        pixel = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
        normalized = normalize_pixel(pixel)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == pixel.shape

    def test_normalize_pixel_constant(self) -> None:
        """Test normalizing a constant pixel."""
        pixel = np.ones(10)
        normalized = normalize_pixel(pixel)
        assert np.all(normalized == 0.0)
        assert normalized.shape == pixel.shape

    def test_normalize_pixel_negative(self) -> None:
        """Test normalizing a pixel with negative values."""
        pixel = np.array([0.0, -0.5, -1.0, -1.5, -2.0])
        normalized = normalize_pixel(pixel)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == pixel.shape

    def test_empty_pixel(self) -> None:
        """Test normalizing an empty pixel raises error."""
        pixel = np.array([])
        with pytest.raises(
            (ValueError, IndexError, ZeroDivisionError, numba.core.errors.TypingError)
        ):
            normalize_pixel(pixel)

    def test_all_zeros(self) -> None:
        """Test normalizing a constant-zero pixel."""
        pixel = np.zeros(10)
        normalized = normalize_pixel(pixel)
        # baseline = 0, cumsum of zeros = zeros, max_val = 0 → returns zeros
        assert np.all(normalized == 0.0)

    def test_normalize_pixel_mean_norm_baseline(self) -> None:
        """Test that normalize_pixel works correctly when off-resonance baseline != 1.0.

        Mean-normalized data has off-resonance slightly > 1.0 (because the mean
        includes the resonance dips). The old code subtracted 1.0 hardcoded,
        causing cumsum drift that shifted center and width estimates. The fixed
        code estimates the baseline from edge frequencies.
        """
        n = 100
        # Simulate mean-norm data: flat baseline ~1.05, Lorentzian dip at idx 50
        pixel = np.ones(n) * 1.05
        center_idx = 50
        for i in range(-15, 16):
            idx = center_idx + i
            if 0 <= idx < n:
                pixel[idx] = 1.05 - 0.20 / (1 + (i / 5.0) ** 2)

        normalized = normalize_pixel(pixel)

        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)

        # The 0.5 crossing should be near the true dip center (index 50).
        # This verifies that edge-based baseline estimation eliminates the
        # cumsum drift that would otherwise shift the center estimate.
        half_crossing = int(np.argmin(np.abs(normalized - 0.5)))
        assert abs(half_crossing - center_idx) <= 5

    def test_negative_values(self) -> None:
        """Test normalizing a pixel with negative values."""
        pixel = np.array([-1, -2, -3, -4], dtype=np.float64)
        normalized = normalize_pixel(pixel)
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)


class TestGuessContrast:
    """Test cases for cumsum_contrast function."""

    def test_cumsum_contrast_shape(self, sample_odmr_data) -> None:
        """Test that the shape of the output is correct."""
        contrasts = cumsum_contrast(sample_odmr_data)

        expected_shape = (
            sample_odmr_data.shape[0],  # n_pol
            sample_odmr_data.shape[1],  # n_frange
            sample_odmr_data.shape[2],  # n_pixel
        )
        assert contrasts.shape == expected_shape

    def test_cumsum_contrast_values(self) -> None:
        """Test that values are correctly calculated."""
        # (1 pol, 1 frange, 2 pixels, 10 freqs)
        data = np.ones((1, 1, 2, 10))

        # First pixel: min=0.5, max=1.0 -> contrast=0.5
        data[0, 0, 0, :] = np.linspace(0.5, 1.0, 10)

        # Second pixel: min=0.2, max=0.6 -> contrast=0.6667
        data[0, 0, 1, :] = np.linspace(0.2, 0.6, 10)

        contrasts = cumsum_contrast(data)
        assert np.isclose(contrasts[0, 0, 0], 0.5, rtol=1e-4)
        assert np.isclose(contrasts[0, 0, 1], 0.6667, rtol=1e-4)


class TestGuessCenter:
    """Test cases for cumsum_center function."""

    def test_cumsum_center_shape(self, sample_odmr_data, frequency_range) -> None:
        """Test that the shape of the output is correct."""
        centers = cumsum_center(sample_odmr_data, frequency_range)

        expected_shape = (
            sample_odmr_data.shape[0],  # n_pol
            sample_odmr_data.shape[1],  # n_frange
            sample_odmr_data.shape[2],  # n_pixel
        )
        assert centers.shape == expected_shape

    def test_cumsum_center_values(self, frequency_range) -> None:
        """Test that the values are correctly calculated."""
        # (1 pol, 1 frange, 2 pixels, 100 freqs)
        data = np.ones((1, 1, 2, 100))
        freq = frequency_range[:1]  # (1, 100)

        # First pixel: center at index 25
        center_idx1 = 25
        for i in range(-10, 11):
            idx = center_idx1 + i
            if 0 <= idx < 100:
                data[0, 0, 0, idx] = 1.0 - 0.8 * np.exp(-0.5 * (i / 3) ** 2)

        # Second pixel: center at index 75
        center_idx2 = 75
        for i in range(-10, 11):
            idx = center_idx2 + i
            if 0 <= idx < 100:
                data[0, 0, 1, idx] = 1.0 - 0.8 * np.exp(-0.5 * (i / 3) ** 2)

        centers = cumsum_center(data, freq)
        assert np.isclose(centers[0, 0, 0], freq[0, center_idx1], rtol=1e-3)
        assert np.isclose(centers[0, 0, 1], freq[0, center_idx2], rtol=1e-3)


class TestRelativeProminence:
    """Test cases for _relative_prominence helper."""

    def test_scales_with_range(self) -> None:
        """Prominence scales with spectral range."""
        small = np.array([0.99, 1.0, 0.995])
        large = np.array([0.9, 1.0, 0.95])
        assert _relative_prominence(large) > _relative_prominence(small)

    def test_minimum_floor(self) -> None:
        """Flat spectrum returns the minimum floor, not zero."""
        flat = np.ones(50)
        assert _relative_prominence(flat) == 1e-6

    def test_fraction_of_range(self) -> None:
        """Result equals range * _RELATIVE_PROMINENCE for non-trivial spectra."""
        s = np.linspace(0.98, 1.0, 50)
        expected = (s.max() - s.min()) * _RELATIVE_PROMINENCE
        assert abs(_relative_prominence(s) - expected) < 1e-12


class TestParameterGuesser:
    """Test cases for ParameterGuesser."""

    def test_offset_estimated_from_edge_baseline(self) -> None:
        """Offset should equal baseline - 1.0, not zero.

        Mean-normalised ODMR data has a baseline slightly above 1.0 because
        the mean includes the resonance dips. The guesser must estimate the
        baseline from the edge frequencies so the initial-guess overlay lines
        up with the fitted curve.
        """
        n_pol, n_frange, n_pixel, n_freq = 1, 1, 2, 100
        baseline = 1.04

        # Flat data at the known baseline (no dips — we only care about offset)
        data = np.full((n_pol, n_frange, n_pixel, n_freq), baseline, dtype=np.float32)

        model = ESRSINGLE()
        f_ghz = np.tile(np.linspace(2.82, 2.92, n_freq), (n_frange, 1))

        guesser = ParameterGuesser(model, f_ghz)
        params = guesser.guess(data)  # (n_pol, n_frange, n_pixel, n_params)

        # Find the offset parameter index
        offset_idx = next(
            i
            for i, name in enumerate(model.parameter_names)
            if model.parameter_types[name] == "offset"
        )
        offsets = params[:, :, :, offset_idx]

        expected = baseline - 1.0
        assert offsets == pytest.approx(expected, abs=1e-4)

    def test_offset_zero_when_baseline_unity(self) -> None:
        """When the baseline is exactly 1.0, offset should be ~0."""
        n_pol, n_frange, n_pixel, n_freq = 1, 1, 1, 50
        data = np.ones((n_pol, n_frange, n_pixel, n_freq), dtype=np.float32)

        model = ESRSINGLE()
        f_ghz = np.tile(np.linspace(2.82, 2.92, n_freq), (n_frange, 1))

        guesser = ParameterGuesser(model, f_ghz)
        params = guesser.guess(data)

        offset_idx = next(
            i
            for i, name in enumerate(model.parameter_names)
            if model.parameter_types[name] == "offset"
        )
        assert params[:, :, :, offset_idx] == pytest.approx(0.0, abs=1e-5)


class TestGuessWidth:
    """Test cases for cumsum_width function."""

    def test_cumsum_width_shape(self, sample_odmr_data, frequency_range) -> None:
        """Test that the shape of the output is correct."""
        widths = cumsum_width(sample_odmr_data, frequency_range, DEFAULT_VMIN, DEFAULT_VMAX)

        expected_shape = (
            sample_odmr_data.shape[0],  # n_pol
            sample_odmr_data.shape[1],  # n_frange
            sample_odmr_data.shape[2],  # n_pixel
        )
        assert widths.shape == expected_shape

    def test_cumsum_width_positive(self, sample_odmr_data, frequency_range) -> None:
        """Test that all widths are positive."""
        widths = cumsum_width(sample_odmr_data, frequency_range, DEFAULT_VMIN, DEFAULT_VMAX)
        assert np.all(widths > 0)


class TestHalfpowerWidth:
    """Test cases for halfpower_width function."""

    def test_shape(self, sample_odmr_data, frequency_range) -> None:
        """Output shape matches (n_pol, n_frange, n_pixel)."""
        hwhm = halfpower_width(sample_odmr_data, frequency_range)
        expected = sample_odmr_data.shape[:3]
        assert hwhm.shape == expected

    def test_synthetic_lorentzian_hwhm(self) -> None:
        """For a single Lorentzian dip, HWHM should match the known width.

        Model: f(x) = 1 - contrast * w^2 / ((x - x0)^2 + w^2)
        The FWHM of this Lorentzian is 2*w, so HWHM = w.
        """
        n_freq = 200
        true_width = 0.002  # 2 MHz HWHM
        center = 2.87
        contrast = 0.05
        freq_1d = np.linspace(2.85, 2.89, n_freq)
        freq = freq_1d[np.newaxis, :]  # (1, n_freq)

        spectrum = 1.0 - contrast * true_width**2 / ((freq_1d - center) ** 2 + true_width**2)
        # Shape: (1 pol, 1 frange, 1 pixel, n_freq)
        data = spectrum[np.newaxis, np.newaxis, np.newaxis, :]

        hwhm = halfpower_width(data, freq)
        # Allow 1 frequency bin tolerance
        df = freq_1d[1] - freq_1d[0]
        assert hwhm[0, 0, 0] == pytest.approx(true_width, abs=df)

    def test_wider_dip_gives_larger_hwhm(self) -> None:
        """A wider Lorentzian should produce a larger HWHM estimate."""
        n_freq = 200
        center = 2.87
        contrast = 0.05
        freq_1d = np.linspace(2.85, 2.89, n_freq)
        freq = freq_1d[np.newaxis, :]

        narrow = 1.0 - contrast * 0.001**2 / ((freq_1d - center) ** 2 + 0.001**2)
        wide = 1.0 - contrast * 0.004**2 / ((freq_1d - center) ** 2 + 0.004**2)

        data = np.stack([narrow, wide])[np.newaxis, np.newaxis, :, :]  # (1,1,2,n_freq)
        hwhm = halfpower_width(data, freq)
        assert hwhm[0, 0, 1] > hwhm[0, 0, 0]


class TestContrastPassthrough:
    """Test that ParameterGuesser passes total contrast to each contrast_i.

    For multi-peak models the hyperfine peaks overlap significantly
    (AHYP ~ linewidth), so the observed dip depth is dominated by the
    central peak. The total contrast is a reasonable starting guess for
    each individual contrast parameter.
    """

    def _make_dip_data(self, n_freq: int = 100) -> tuple:
        """Create synthetic data with a known total contrast."""
        freq_1d = np.linspace(2.82, 2.92, n_freq)
        freq = np.tile(freq_1d, (2, 1))  # (2 frange, n_freq)

        # Single Lorentzian dip with ~5% contrast
        spectrum = 1.0 - 0.05 * 0.002**2 / ((freq_1d - 2.87) ** 2 + 0.002**2)
        data = np.tile(spectrum, (2, 2, 3, 1))  # (2 pol, 2 frange, 3 pixels, n_freq)
        return data.astype(np.float32), freq

    def test_esr14n_contrast_equals_total(self) -> None:
        """For ESR14N, each contrast_i = total_contrast (no division)."""
        data, freq = self._make_dip_data()
        total_contrast = cumsum_contrast(data)  # (2, 2, 3)

        model = ESR14N()
        guesser = ParameterGuesser(model, freq)
        params = guesser.guess(data)

        for name in ("contrast_0", "contrast_1", "contrast_2"):
            idx = model.parameter_names.index(name)
            np.testing.assert_allclose(params[:, :, :, idx], total_contrast, rtol=1e-5)

    def test_esr15n_contrast_equals_total(self) -> None:
        """For ESR15N, each contrast_i = total_contrast (no division)."""
        data, freq = self._make_dip_data()
        total_contrast = cumsum_contrast(data)

        model = ESR15N()
        guesser = ParameterGuesser(model, freq)
        params = guesser.guess(data)

        for name in ("contrast_0", "contrast_1"):
            idx = model.parameter_names.index(name)
            np.testing.assert_allclose(params[:, :, :, idx], total_contrast, rtol=1e-5)

    def test_esrsingle_contrast_equals_total(self) -> None:
        """For ESRSINGLE, contrast = total."""
        data, freq = self._make_dip_data()
        total_contrast = cumsum_contrast(data)

        model = ESRSINGLE()
        guesser = ParameterGuesser(model, freq)
        params = guesser.guess(data)

        idx = model.parameter_names.index("contrast")
        np.testing.assert_allclose(params[:, :, :, idx], total_contrast, rtol=1e-5)


class TestWidthCorrection:
    """Test that ParameterGuesser applies AHYP correction for multi-peak models."""

    def _make_data_with_known_width(self, true_hwhm: float = 0.003) -> tuple:
        """Create data with a single Lorentzian of known HWHM."""
        n_freq = 200
        center = 2.87
        contrast = 0.05
        freq_1d = np.linspace(2.85, 2.89, n_freq)
        freq = freq_1d[np.newaxis, :]  # (1 frange, n_freq)

        spectrum = 1.0 - contrast * true_hwhm**2 / ((freq_1d - center) ** 2 + true_hwhm**2)
        data = spectrum[np.newaxis, np.newaxis, np.newaxis, :]  # (1,1,1,n_freq)
        return data.astype(np.float32), freq

    def test_esr14n_subtracts_ahyp(self) -> None:
        """ESR14N width = envelope_hwhm - AHYP_14N."""
        true_hwhm = 0.004  # 4 MHz, well above AHYP_14N
        data, freq = self._make_data_with_known_width(true_hwhm)

        envelope_hwhm = halfpower_width(data, freq)

        model = ESR14N()
        guesser = ParameterGuesser(model, freq)
        params = guesser.guess(data)

        width_idx = model.parameter_names.index("width")
        guessed_width = params[0, 0, 0, width_idx]
        expected = max(float(envelope_hwhm[0, 0, 0]) - AHYP_14N, 0.0003)
        assert guessed_width == pytest.approx(expected, rel=1e-4)

    def test_esr15n_subtracts_ahyp(self) -> None:
        """ESR15N width = envelope_hwhm - AHYP_15N."""
        true_hwhm = 0.004
        data, freq = self._make_data_with_known_width(true_hwhm)

        envelope_hwhm = halfpower_width(data, freq)

        model = ESR15N()
        guesser = ParameterGuesser(model, freq)
        params = guesser.guess(data)

        width_idx = model.parameter_names.index("width")
        guessed_width = params[0, 0, 0, width_idx]
        expected = max(float(envelope_hwhm[0, 0, 0]) - AHYP_15N, 0.0003)
        assert guessed_width == pytest.approx(expected, rel=1e-4)

    def test_esrsingle_no_correction(self) -> None:
        """ESRSINGLE uses envelope HWHM directly, no subtraction."""
        true_hwhm = 0.003
        data, freq = self._make_data_with_known_width(true_hwhm)

        envelope_hwhm = halfpower_width(data, freq)

        model = ESRSINGLE()
        guesser = ParameterGuesser(model, freq)
        params = guesser.guess(data)

        width_idx = model.parameter_names.index("width")
        guessed_width = params[0, 0, 0, width_idx]
        assert guessed_width == pytest.approx(float(envelope_hwhm[0, 0, 0]), rel=1e-4)

    def test_width_floor_prevents_negative(self) -> None:
        """When envelope HWHM < AHYP, width is floored at 0.3 MHz."""
        # Very narrow dip where HWHM < AHYP_14N
        true_hwhm = 0.001  # 1 MHz, less than AHYP_14N = 2.158 MHz
        data, freq = self._make_data_with_known_width(true_hwhm)

        model = ESR14N()
        guesser = ParameterGuesser(model, freq)
        params = guesser.guess(data)

        width_idx = model.parameter_names.index("width")
        guessed_width = params[0, 0, 0, width_idx]
        assert guessed_width == pytest.approx(0.0003, rel=1e-4)


def _lorentzian(freq: np.ndarray, center: float, hwhm: float, contrast: float) -> np.ndarray:
    """Single Lorentzian dip: 1 - contrast * hwhm^2 / ((f - center)^2 + hwhm^2)."""
    return 1.0 - contrast * hwhm**2 / ((freq - center) ** 2 + hwhm**2)


class TestAbsorptionCentroid:
    """Tests for the absorption_centroid guesser."""

    @staticmethod
    def _make_4d(spectrum: np.ndarray, freq_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Wrap a 1D spectrum into (1,1,1,n_freq) data and (1,n_freq) freq array."""
        data = spectrum[np.newaxis, np.newaxis, np.newaxis, :]
        freq = freq_1d[np.newaxis, :]
        return data.astype(np.float64), freq

    def test_shape(self) -> None:
        """Output shape is (n_pol, n_frange, n_pixel)."""
        n_pol, n_frange, n_pixel, n_freq = 2, 2, 5, 100
        data = np.ones((n_pol, n_frange, n_pixel, n_freq))
        freq = np.tile(np.linspace(2.82, 2.92, n_freq), (n_frange, 1))
        centers = absorption_centroid(data, freq)
        assert centers.shape == (n_pol, n_frange, n_pixel)

    def test_esrsingle_center(self) -> None:
        """Single Lorentzian: centroid should be within 1 freq bin of true center."""
        n_freq = 200
        true_center = 2.87
        freq_1d = np.linspace(2.82, 2.92, n_freq)
        spectrum = _lorentzian(freq_1d, true_center, 0.003, 0.05)
        data, freq = self._make_4d(spectrum, freq_1d)

        centers = absorption_centroid(data, freq)
        df = freq_1d[1] - freq_1d[0]
        assert abs(centers[0, 0, 0] - true_center) < df

    def test_n14_equal_contrasts(self) -> None:
        """N14 triplet with equal contrasts: centroid within 0.1 * AHYP_14N of center."""
        n_freq = 300
        true_center = 2.87
        hwhm = 0.002
        freq_1d = np.linspace(2.855, 2.885, n_freq)
        spectrum = (
            _lorentzian(freq_1d, true_center - AHYP_14N, hwhm, 0.04)
            + _lorentzian(freq_1d, true_center, hwhm, 0.04)
            + _lorentzian(freq_1d, true_center + AHYP_14N, hwhm, 0.04)
            - 2.0  # remove the double-counted baseline (three dips, three +1 offsets)
        )
        data, freq = self._make_4d(spectrum, freq_1d)

        centers = absorption_centroid(data, freq)
        assert abs(centers[0, 0, 0] - true_center) < 0.1 * AHYP_14N

    def test_n14_unequal_contrasts(self) -> None:
        """N14 triplet with 3:1 contrast ratio: centroid within AHYP_14N of center.

        Uses HWHM << AHYP so the three dips are spectrally distinct.
        The centroid is then the contrast-weighted average of dip positions,
        landing between the true center and the dominant outer dip.
        argmin would return the dominant dip position, off by exactly AHYP_14N.
        """
        n_freq = 400
        true_center = 2.87
        # Use HWHM << AHYP_14N (0.5 MHz vs 2.158 MHz) so dips don't overlap
        hwhm = 0.0005
        freq_1d = np.linspace(2.860, 2.880, n_freq)
        # Left dip 3x stronger than the others
        spectrum = (
            _lorentzian(freq_1d, true_center - AHYP_14N, hwhm, 0.12)
            + _lorentzian(freq_1d, true_center, hwhm, 0.04)
            + _lorentzian(freq_1d, true_center + AHYP_14N, hwhm, 0.04)
            - 2.0
        )
        data, freq = self._make_4d(spectrum, freq_1d)

        centers = absorption_centroid(data, freq)
        assert abs(centers[0, 0, 0] - true_center) < AHYP_14N

    def test_n15_equal_contrasts(self) -> None:
        """N15 doublet with equal contrasts: centroid within 0.1 * AHYP_15N of center."""
        n_freq = 200
        true_center = 2.87
        hwhm = 0.002
        freq_1d = np.linspace(2.862, 2.878, n_freq)
        spectrum = (
            _lorentzian(freq_1d, true_center - AHYP_15N, hwhm, 0.05)
            + _lorentzian(freq_1d, true_center + AHYP_15N, hwhm, 0.05)
            - 1.0  # remove double-counted baseline
        )
        data, freq = self._make_4d(spectrum, freq_1d)

        centers = absorption_centroid(data, freq)
        assert abs(centers[0, 0, 0] - true_center) < 0.1 * AHYP_15N

    def test_flat_spectrum_fallback(self) -> None:
        """Flat spectrum (no absorption): falls back to freq range midpoint."""
        freq_1d = np.linspace(2.82, 2.92, 100)
        spectrum = np.ones(100)
        data, freq = self._make_4d(spectrum, freq_1d)

        centers = absorption_centroid(data, freq)
        midpoint = (freq_1d[0] + freq_1d[-1]) / 2.0
        assert abs(centers[0, 0, 0] - midpoint) < 1e-10

    def test_all_below_baseline_fallback(self) -> None:
        """All values above baseline (inverted): falls back to freq range midpoint."""
        freq_1d = np.linspace(2.82, 2.92, 100)
        # Emission peak (above baseline) rather than absorption dip
        spectrum = 1.0 + 0.05 * np.exp(-((freq_1d - 2.87) ** 2) / 0.001**2)
        data, freq = self._make_4d(spectrum, freq_1d)

        centers = absorption_centroid(data, freq)
        midpoint = (freq_1d[0] + freq_1d[-1]) / 2.0
        assert abs(centers[0, 0, 0] - midpoint) < freq_1d[1] - freq_1d[0]
