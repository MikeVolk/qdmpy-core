"""Shared test fixtures for QDMpy test suite.

This module provides common fixtures used across unit tests, eliminating
duplication and ensuring consistent test data across the project.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from qdmpy_core.fitting.result import FitResult
from qdmpy_core.settings import (
    FitSettings,
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
)

# ============================================================================
# Configuration
# ============================================================================

# Mock settings for tests (all frequencies in GHz)
MOCK_SETTINGS = QDMpySettings(
    fit=FitSettings(
        estimator='LSE',
        max_number_iterations=100,
        tolerance=1e-6,
    ),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            center_min=2.8,
            center_max=2.9,
            center_type='FREE',
            width_min=0.001,
            width_max=0.01,
            width_type='FREE',
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type='FREE',
            offset_min=-0.1,
            offset_max=0.1,
            offset_type='FREE',
        )
    ),
)


# ============================================================================
# Helpers
# ============================================================================

def make_xr_data(numpy_4d: np.ndarray) -> xr.DataArray:
    """Convert 4D numpy (n_pol, n_frange, n_pixel, n_freq) to 5D xr.DataArray.

    Args:
        numpy_4d: 4D array with shape (n_pol, n_frange, n_pixel, n_freq)

    Returns:
        5D xr.DataArray with dims (polarity, freq_range, y, x, freq_idx)

    Raises:
        AssertionError: If n_pixel is not a perfect square.
    """
    n_pol, n_frange, n_pixel, n_freq = numpy_4d.shape
    side = int(np.sqrt(n_pixel))
    assert side * side == n_pixel, f'n_pixel={n_pixel} is not a perfect square'

    data_5d = numpy_4d.reshape(n_pol, n_frange, side, side, n_freq)
    freq_ghz = np.tile(np.linspace(2.87, 2.88, n_freq), (n_frange, 1))

    return xr.DataArray(
        data_5d,
        dims=('polarity', 'freq_range', 'y', 'x', 'freq_idx'),
        coords={
            'polarity': ['neg', 'pos'][:n_pol],
            'freq_range': ['low', 'high'][:n_frange],
            'freq_ghz': (('freq_range', 'freq_idx'), freq_ghz),
        },
    )


# ============================================================================
# Basic Fixtures
# ============================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_numpy_data() -> np.ndarray:
    """Create sample 4D numpy data (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        Array of shape (2, 2, 4, 10) with random data.
    """
    data = np.ones((2, 2, 4, 10))

    for pol in range(2):
        for frange in range(2):
            # Add spectral structure: narrow dip
            center_freq_idx = 5
            width = 1.5
            for pixel in range(4):
                spectrum = np.ones(10)
                for i in range(10):
                    dist = (i - center_freq_idx) / width
                    spectrum[i] = 1.0 - 0.5 * np.exp(-(dist**2) / 2)
                data[pol, frange, pixel, :] = spectrum

    return data


@pytest.fixture
def sample_frequencies() -> np.ndarray:
    """Create sample frequencies for testing.

    Returns:
        1D array of 10 frequencies in GHz.
    """
    return np.linspace(2.87, 2.88, 10)


# ============================================================================
# xarray and ODMR Fixtures
# ============================================================================

@pytest.fixture
def sample_data(sample_numpy_data: np.ndarray) -> tuple[xr.DataArray, tuple[int, int], np.ndarray]:
    """Create sample xr.DataArray, scan dimensions, and frequencies.

    Args:
        sample_numpy_data: 4D array from fixture

    Returns:
        Tuple of (xr.DataArray, scan_dimensions, frequencies)
    """
    data_xr = make_xr_data(sample_numpy_data)
    scan_dimensions = (2, 2)  # sqrt(4)
    frequencies = np.tile(np.linspace(2.87, 2.88, 10), (2, 1))
    return data_xr, scan_dimensions, frequencies


# ============================================================================
# FitResult Fixtures
# ============================================================================

@pytest.fixture
def sample_parameters() -> dict[str, np.ndarray]:
    """Create sample fit parameters for FitResult testing.

    Returns:
        Dictionary with center, width, contrast, offset, chi2, states arrays.
    """
    n_pixels = 100
    return {
        'center': np.random.normal(2.87, 0.001, n_pixels),  # ~2.87 GHz
        'width_0': np.random.normal(0.0005, 0.00001, n_pixels),  # ~0.5 MHz in GHz
        'contrast': np.random.uniform(0.01, 0.1, n_pixels),  # 1-10% contrast
        'offset': np.random.normal(0, 0.01, n_pixels),  # Small offsets
        'chi2': np.random.exponential(1.0, n_pixels),  # Chi-squared values
        'states': np.random.choice([0, 1], n_pixels, p=[0.9, 0.1]),  # 90% convergence
    }


@pytest.fixture
def sample_fit_result(sample_parameters: dict[str, np.ndarray]) -> FitResult:
    """Create a sample FitResult instance for testing.

    Args:
        sample_parameters: Parameter dictionary from fixture

    Returns:
        FitResult instance with test parameters.
    """
    return FitResult(
        parameters=sample_parameters,
        scan_dimensions=(10, 10),
        pixel_spacing=4e-6,
        model_name='ESR15N',
        metadata={'test': True, 'quality_metrics': {'mean_chi2': 1.0}},
    )
