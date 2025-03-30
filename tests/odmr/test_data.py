"""Test module for QDMpy.odmr.data
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.io import MatlabLoader


@pytest.fixture
def sample_data() -> tuple[NDArray, NDArray, NDArray]:
    """Provide sample data for testing."""
    # Create sample ODMR data
    data = np.random.random((2, 3, 100, 50))  # (modes, reps, pixels, frequencies)
    scan_dimensions = np.array([10, 10])  # 10x10 grid
    frequencies = np.linspace(2.87e9, 2.89e9, 50)  # 50 frequencies
    return data, scan_dimensions, frequencies


@pytest.fixture
def sample_odmr_data(sample_data) -> ODMRData:
    """Provide sample ODMRData instance."""
    data, scan_dimensions, frequencies = sample_data
    return ODMRData(data, scan_dimensions, frequencies)


@pytest.fixture
def matlab_loader() -> MatlabLoader:
    """Provide a MatlabLoader instance for testing."""
    test_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    return MatlabLoader(data_folder=test_data_path)


class TestODMRData:
    """Test class for ODMRData."""

    def test_init(self, sample_data):
        """Test initialization with standard parameters."""
        data, scan_dimensions, frequencies = sample_data
        odmr_data = ODMRData(data, scan_dimensions, frequencies)

        assert odmr_data.data is data
        assert odmr_data.scan_dimensions is scan_dimensions
        assert odmr_data.frequencies is frequencies
        assert isinstance(odmr_data.metadata, dict)
        assert len(odmr_data.metadata) == 0

    def test_init_with_metadata(self, sample_data):
        """Test initialization with metadata."""
        data, scan_dimensions, frequencies = sample_data
        metadata = {'test_key': 'test_value'}
        odmr_data = ODMRData(data, scan_dimensions, frequencies, metadata)

        assert odmr_data.metadata == metadata

    def test_shape_property(self, sample_odmr_data):
        """Test the shape property."""
        assert sample_odmr_data.shape == sample_odmr_data.data.shape

    def test_from_loader(self, matlab_loader):
        """Test creation from a loader."""
        odmr_data = ODMRData.from_loader(matlab_loader)

        assert isinstance(odmr_data, ODMRData)
        assert isinstance(odmr_data.data, np.ndarray)
        assert isinstance(odmr_data.scan_dimensions, np.ndarray)
        assert isinstance(odmr_data.frequencies, np.ndarray)

    def test_from_loader_error(self):
        """Test error handling in from_loader method."""
        class FailingLoader:
            def load(self, **kwargs):
                raise ValueError('Test error')

        with pytest.raises(RuntimeError):
            ODMRData.from_loader(FailingLoader())
