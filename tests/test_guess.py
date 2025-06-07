"""Test module for QDMpy.guess

These tests cover the model and parameter guessing functionality used for ODMR data
processing and fitting.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We'll use the original numba functions directly since we've fixed them
# in the main codebase by simplifying the decorators and fixing the logic

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the module to test
from QDMpy.constants import DEFAULT_VMAX, DEFAULT_VMIN
from QDMpy.exceptions import ModelGuessNotPossible
from QDMpy.guess import (
    get_model_by_peaks,
    guess_center,
    guess_center_pixel,
    guess_contrast,
    guess_contrast_pixel,
    guess_initial_fit_parameters,
    guess_model,
    guess_n_peaks,
    guess_width,
    guess_width_pixel,
    normalize_pixel,
    validate_array,
)
from QDMpy.models import ESR14N, ESR15N, ESRSINGLE


@pytest.fixture
def sample_odmr_data():
    """Create a sample ODMR data array for testing."""
    # Create a 4D array with 2 polarities, 3 frequency ranges, 100 frequencies, and 10 pixels
    data = np.random.random((2, 3, 100, 10))

    # Add some peaks to make it more realistic for peak detection
    for pol in range(2):
        for freq_range in range(3):
            # Create a dip around 1/3 of the way through the frequencies
            center_idx1 = 30
            for pixel in range(10):
                for i in range(-5, 6):
                    if 0 <= center_idx1 + i < 100:
                        data[pol, freq_range, center_idx1 + i, pixel] = 0.5 - 0.4 * np.exp(-0.5 * (i / 2)**2)

            # For ESR15N, create a second dip
            center_idx2 = 70
            for pixel in range(10):
                for i in range(-5, 6):
                    if 0 <= center_idx2 + i < 100:
                        data[pol, freq_range, center_idx2 + i, pixel] = 0.5 - 0.4 * np.exp(-0.5 * (i / 2)**2)

    return data


@pytest.fixture
def frequency_range():
    """Create a sample frequency range for testing."""
    return np.linspace(2.87e9, 2.89e9, 100)


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

    def test_validate_correct_dimensions(self):
        """Test validation with correct dimensions."""
        data = np.zeros((2, 3, 4, 5))
        # Should not raise an exception
        validate_array(data, 4, 'test_data')

    def test_validate_incorrect_dimensions(self):
        """Test validation with incorrect dimensions."""
        data = np.zeros((2, 3, 4))
        # Should raise a ValueError
        with pytest.raises(ValueError) as excinfo:
            validate_array(data, 4, 'test_data')
        assert 'must have 4 dimensions' in str(excinfo.value)
        assert 'Got 3' in str(excinfo.value)

    def test_none_array(self):
        """Test validation with None."""
        with pytest.raises(ValueError):
            validate_array(None, 4, 'test_data')

    def test_non_numeric_array(self):
        """Test validation with a non-numeric array."""
        data = np.array([['a', 'b'], ['c', 'd']])
        with pytest.raises(ValueError):
            validate_array(data, 2, 'test_data')

    def test_unexpected_dimensions(self):
        """Test validation with unexpected dimensions."""
        data = np.zeros((2, 3, 4))
        with pytest.raises(ValueError):
            validate_array(data, 5, 'test_data')


class TestGuessNPeaks:
    """Test cases for guess_n_peaks function."""

    def test_guess_n_peaks_mock(self):
        """Test guessing the number of peaks with mocked find_peaks function."""
        # Create a mock data array
        mock_data = np.random.random((2, 3, 100, 10))

        # Mock the find_peaks function to return consistent results
        with patch('QDMpy.guess.find_peaks') as mock_find_peaks:
            # Set up the mock to return 2 peaks for every call
            mock_find_peaks.return_value = (np.array([30, 70]), {})

            # Call guess_n_peaks with the mock in place
            n_peaks, doubt, indices = guess_n_peaks(mock_data)

            # Since we're returning 2 peaks consistently, doubt should be False
            assert n_peaks == 2
            assert bool(doubt) is False
            assert len(indices) == mock_data.shape[0] * mock_data.shape[1]

    def test_guess_n_peaks_doubt(self):
        """Test guessing the number of peaks when there's doubt (inconsistent peak counts)."""
        # Create a mock data array
        mock_data = np.random.random((2, 3, 100, 10))

        # Mock the find_peaks function to return inconsistent results
        with patch('QDMpy.guess.find_peaks') as mock_find_peaks:
            # Set up the mock to return different numbers of peaks
            # First polarity, first freq range: 2 peaks
            # First polarity, second freq range: 2 peaks
            # First polarity, third freq range: 3 peaks (different!)
            # Second polarity, all freq ranges: 2 peaks
            def side_effect_fn(data, prominence):
                # Get the current indices based on mock_find_peaks.call_count
                call_count = mock_find_peaks.call_count - 1  # 0-indexed

                if call_count == 2:  # First polarity, third freq range
                    return np.array([25, 50, 75]), {}
                return np.array([30, 70]), {}

            mock_find_peaks.side_effect = side_effect_fn

            # Call guess_n_peaks with the mock in place
            n_peaks, doubt, indices = guess_n_peaks(mock_data)

            # The average should be close to 2, but doubt should be True
            # The function should still return a rounded integer
            assert n_peaks in (2, 3)  # Depending on rounding
            assert bool(doubt) is True  # There should be doubt due to inconsistency
            assert len(indices) == mock_data.shape[0] * mock_data.shape[1]

    def test_incorrect_dimensions(self):
        """Test with incorrect dimensions."""
        data = np.zeros((2, 3, 4))  # 3D array instead of 4D
        with pytest.raises(ValueError):
            guess_n_peaks(data)


class TestGetModelByPeaks:
    """Test cases for get_model_by_peaks function."""

    def test_get_model_single_peak(self, model_instances):
        """Test getting the model for a single peak."""
        model = get_model_by_peaks(1)
        assert isinstance(model, ESRSINGLE)
        assert model.n_peaks == 1
        assert model.name == 'ESRSINGLE'

    def test_get_model_two_peaks(self, model_instances):
        """Test getting the model for two peaks."""
        model = get_model_by_peaks(2)
        assert isinstance(model, ESR15N)
        assert model.n_peaks == 2
        assert model.name == 'ESR15N'

    def test_get_model_three_peaks(self, model_instances):
        """Test getting the model for three peaks."""
        model = get_model_by_peaks(3)
        assert isinstance(model, ESR14N)
        assert model.n_peaks == 3
        assert model.name == 'ESR14N'

    def test_get_model_invalid_peaks(self):
        """Test with an invalid number of peaks."""
        with pytest.raises(ValueError) as excinfo:
            get_model_by_peaks(4)  # No model registered with 4 peaks
        assert 'No model found for 4 peaks' in str(excinfo.value)


class TestGuessModel:
    """Test cases for guess_model function."""

    @patch('QDMpy.guess.guess_n_peaks')
    def test_guess_model_no_doubt(self, mock_guess_n_peaks):
        """Test guessing model when there's no doubt."""
        # Mock the guess_n_peaks function to return no doubt
        mock_guess_n_peaks.return_value = (2, False, [])

        # Create some dummy data
        data = np.zeros((2, 3, 100, 10))

        # Call the function
        model = guess_model(data)

        # Check the result
        assert isinstance(model, ESR15N)
        assert model.n_peaks == 2

    @patch('QDMpy.guess.guess_n_peaks')
    def test_guess_model_with_doubt(self, mock_guess_n_peaks):
        """Test guessing model when there's doubt."""
        # Mock the guess_n_peaks function to return doubt
        mock_guess_n_peaks.return_value = (2, True, [])

        # Create some dummy data
        data = np.zeros((2, 3, 100, 10))

        # Call the function should raise an exception
        with pytest.raises(ModelGuessNotPossible):
            guess_model(data)


class TestNormalizePixel:
    """Test cases for normalize_pixel function."""

    def test_normalize_pixel_normal(self):
        """Test normalizing a pixel with normal data."""
        # Create a sample pixel
        pixel = np.array([1.0, 1.2, 1.5, 1.8, 2.0])

        # Normalize it using the original function
        normalized = normalize_pixel(pixel)

        # Check the result
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == pixel.shape

    def test_normalize_pixel_constant(self):
        """Test normalizing a constant pixel."""
        # Create a constant pixel
        pixel = np.ones(10)

        # Normalize it using the original function
        normalized = normalize_pixel(pixel)

        # Because there's no variation, should be all zeros
        assert np.all(normalized == 0.0)
        assert normalized.shape == pixel.shape

    def test_normalize_pixel_negative(self):
        """Test normalizing a pixel with negative values."""
        # Create a pixel with negative values
        pixel = np.array([0.0, -0.5, -1.0, -1.5, -2.0])

        # Normalize it using the original function
        normalized = normalize_pixel(pixel)

        # Check the result
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == pixel.shape

    def test_empty_pixel(self):
        """Test normalizing an empty pixel."""
        pixel = np.array([])
        normalized = normalize_pixel(pixel)
        assert normalized.size == 0

    def test_all_zeros(self):
        """Test normalizing a pixel with all zeros."""
        pixel = np.zeros(10)
        normalized = normalize_pixel(pixel)
        assert np.all(normalized == 0.0)

    def test_negative_values(self):
        """Test normalizing a pixel with negative values."""
        pixel = np.array([-1, -2, -3, -4])
        normalized = normalize_pixel(pixel)
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)


class TestGuessContrastPixel:
    """Test cases for guess_contrast_pixel function."""

    def test_guess_contrast_pixel_normal(self):
        """Test guessing contrast with normal data."""
        # Create a sample pixel with known min and max
        pixel = np.array([0.5, 0.7, 0.3, 0.9, 0.6])

        # Calculate contrast using the original function
        contrast = guess_contrast_pixel(pixel)

        # Expected: (0.9 - 0.3) / 0.9 = 0.6667
        assert np.isclose(contrast, 0.6667, rtol=1e-4)

    def test_guess_contrast_pixel_zero(self):
        """Test guessing contrast with zero values."""
        # Create a pixel with a max value of 0
        pixel = np.zeros(5)

        # Calculate contrast using the original function
        contrast = guess_contrast_pixel(pixel)

        # Should return 0
        assert contrast == 0.0

    def test_guess_contrast_pixel_constant(self):
        """Test guessing contrast with constant values."""
        # Create a constant non-zero pixel
        pixel = np.ones(5)

        # Calculate contrast using the original function
        contrast = guess_contrast_pixel(pixel)

        # With no variation, should be 0
        assert contrast == 0.0


class TestGuessContrast:
    """Test cases for guess_contrast function."""

    def test_guess_contrast_shape(self, sample_odmr_data):
        """Test that the shape of the output is correct."""
        # Calculate contrasts
        contrasts = guess_contrast(sample_odmr_data)

        # Check shape
        expected_shape = (
            sample_odmr_data.shape[0],  # n_polarities
            sample_odmr_data.shape[1],  # n_freq_ranges
            sample_odmr_data.shape[3],  # n_pixels
        )
        assert contrasts.shape == expected_shape

    def test_guess_contrast_values(self):
        """Test that the values are correctly calculated."""
        # Create a simple data set with known contrasts
        data = np.ones((1, 1, 10, 2))  # 1 polarity, 1 freq range, 10 freqs, 2 pixels

        # First pixel: min=0.5, max=1.0 -> contrast=0.5
        data[0, 0, :, 0] = np.linspace(0.5, 1.0, 10)

        # Second pixel: min=0.2, max=0.6 -> contrast=0.6667
        data[0, 0, :, 1] = np.linspace(0.2, 0.6, 10)

        # Calculate contrasts
        contrasts = guess_contrast(data)

        # Check values
        assert np.isclose(contrasts[0, 0, 0], 0.5, rtol=1e-4)
        assert np.isclose(contrasts[0, 0, 1], 0.6667, rtol=1e-4)


class TestGuessCenterPixel:
    """Test cases for guess_center_pixel function."""

    def test_guess_center_pixel(self, frequency_range):
        """Test guessing the center frequency of a pixel."""
        # Create a pixel with a clear center
        pixel = np.ones(100)
        # Create a dip in the middle
        center_idx = 50
        for i in range(-10, 11):
            idx = center_idx + i
            if 0 <= idx < 100:
                pixel[idx] = 1.0 - 0.8 * np.exp(-0.5 * (i / 3)**2)

        # Guess the center using the original function
        center = guess_center_pixel(pixel, frequency_range)

        # Should be close to the center frequency
        expected_freq = frequency_range[center_idx]
        assert np.isclose(center, expected_freq, rtol=1e-3)


class TestGuessCenter:
    """Test cases for guess_center function."""

    def test_guess_center_shape(self, sample_odmr_data, frequency_range):
        """Test that the shape of the output is correct."""
        # Calculate centers
        centers = guess_center(sample_odmr_data, frequency_range)

        # Check shape
        expected_shape = (
            sample_odmr_data.shape[0],  # n_polarities
            sample_odmr_data.shape[1],  # n_freq_ranges
            sample_odmr_data.shape[3],  # n_pixels
        )
        assert centers.shape == expected_shape

    def test_guess_center_values(self, frequency_range):
        """Test that the values are correctly calculated."""
        # Create a simple data set with known centers
        data = np.ones((1, 1, 100, 2))  # 1 polarity, 1 freq range, 100 freqs, 2 pixels

        # First pixel: center at index 25
        center_idx1 = 25
        for i in range(-10, 11):
            idx = center_idx1 + i
            if 0 <= idx < 100:
                data[0, 0, idx, 0] = 1.0 - 0.8 * np.exp(-0.5 * (i / 3)**2)

        # Second pixel: center at index 75
        center_idx2 = 75
        for i in range(-10, 11):
            idx = center_idx2 + i
            if 0 <= idx < 100:
                data[0, 0, idx, 1] = 1.0 - 0.8 * np.exp(-0.5 * (i / 3)**2)

        # Calculate centers
        centers = guess_center(data, frequency_range)

        # Check values - should be close to the expected frequencies
        assert np.isclose(centers[0, 0, 0], frequency_range[center_idx1], rtol=1e-3)
        assert np.isclose(centers[0, 0, 1], frequency_range[center_idx2], rtol=1e-3)


class TestGuessWidthPixel:
    """Test cases for guess_width_pixel function."""

    def test_guess_width_pixel(self, frequency_range):
        """Test guessing the width of a pixel."""
        # Create a pixel with a clear center
        pixel = np.ones(100)
        # Create a dip in the middle
        center_idx = 50
        for i in range(-10, 11):
            idx = center_idx + i
            if 0 <= idx < 100:
                pixel[idx] = 1.0 - 0.8 * np.exp(-0.5 * (i / 3)**2)

        # Guess the width using the original function
        width = guess_width_pixel(pixel, frequency_range, DEFAULT_VMIN, DEFAULT_VMAX)

        # Should be a positive value
        assert width > 0

        # The width should be smaller than the full frequency range
        assert width < (frequency_range[-1] - frequency_range[0])


class TestGuessWidth:
    """Test cases for guess_width function."""

    def test_guess_width_shape(self, sample_odmr_data, frequency_range):
        """Test that the shape of the output is correct."""
        # Calculate widths
        widths = guess_width(sample_odmr_data, frequency_range, DEFAULT_VMIN, DEFAULT_VMAX)

        # Check shape
        expected_shape = (
            sample_odmr_data.shape[0],  # n_polarities
            sample_odmr_data.shape[1],  # n_freq_ranges
            sample_odmr_data.shape[3],  # n_pixels
        )
        assert widths.shape == expected_shape

    def test_guess_width_positive(self, sample_odmr_data, frequency_range):
        """Test that all widths are positive."""
        # Calculate widths
        widths = guess_width(sample_odmr_data, frequency_range, DEFAULT_VMIN, DEFAULT_VMAX)

        # All values should be positive
        assert np.all(widths > 0)


class TestGuessInitialFitParameters:
    """Test cases for guess_initial_fit_parameters function."""

    def test_guess_initial_fit_parameters_single(self, sample_odmr_data, frequency_range):
        """Test guessing parameters for ESRSINGLE model."""
        # Get the ESRSINGLE model
        model = ESRSINGLE()

        # Guess parameters
        parameters = guess_initial_fit_parameters(sample_odmr_data, frequency_range, model)

        # Check shape
        expected_shape = (
            sample_odmr_data.shape[0],  # n_polarities
            sample_odmr_data.shape[1],  # n_freq_ranges
            sample_odmr_data.shape[3],  # n_pixels
            len(model.parameters_unique),  # n_parameters
        )
        assert parameters.shape == expected_shape

    def test_guess_initial_fit_parameters_14n(self, sample_odmr_data, frequency_range):
        """Test guessing parameters for ESR14N model."""
        # Get the ESR14N model
        model = ESR14N()

        # Guess parameters
        parameters = guess_initial_fit_parameters(sample_odmr_data, frequency_range, model)

        # Check shape
        expected_shape = (
            sample_odmr_data.shape[0],  # n_polarities
            sample_odmr_data.shape[1],  # n_freq_ranges
            sample_odmr_data.shape[3],  # n_pixels
            len(model.parameters_unique),  # n_parameters
        )
        assert parameters.shape == expected_shape

    def test_guess_initial_fit_parameters_15n(self, sample_odmr_data, frequency_range):
        """Test guessing parameters for ESR15N model."""
        # Get the ESR15N model
        model = ESR15N()

        # Guess parameters
        parameters = guess_initial_fit_parameters(sample_odmr_data, frequency_range, model)

        # Check shape
        expected_shape = (
            sample_odmr_data.shape[0],  # n_polarities
            sample_odmr_data.shape[1],  # n_freq_ranges
            sample_odmr_data.shape[3],  # n_pixels
            len(model.parameters_unique),  # n_parameters
        )
        assert parameters.shape == expected_shape

    def test_guess_initial_fit_parameters_invalid_param(self, sample_odmr_data, frequency_range):
        """Test guessing parameters with an invalid parameter type."""
        # Create a mock model with an unsupported parameter
        mock_model = MagicMock()
        mock_model.parameters_unique = ['invalid_param']

        # Should raise a ValueError
        with pytest.raises(ValueError) as excinfo:
            guess_initial_fit_parameters(sample_odmr_data, frequency_range, mock_model)

        assert "Parameter 'invalid_param' has no defined guess method" in str(excinfo.value)
