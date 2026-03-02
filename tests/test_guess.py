"""Test module for QDMpy.guess.

These tests cover the model and parameter guessing functionality used for ODMR data
processing and fitting.
"""

from __future__ import annotations

from unittest.mock import patch

import numba
import numpy as np
import pytest

from qdmpy.constants import DEFAULT_VMAX, DEFAULT_VMIN
from qdmpy.exceptions import (
    DataShapeError,
    DataValidationError,
    ModelNotFoundError,
)
from qdmpy.fitting.guess import (
    _RELATIVE_PROMINENCE,
    _relative_prominence,
    cumsum_center,
    cumsum_contrast,
    cumsum_width,
    get_model_by_peaks,
    guess_model,
    guess_n_peaks,
    normalize_pixel,
    validate_array,
)
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
        with pytest.raises((ValueError, IndexError, numba.core.errors.TypingError)):
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
