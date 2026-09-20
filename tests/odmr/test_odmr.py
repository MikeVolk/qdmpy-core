"""Test module for QDMpy.odmr.odmr."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from qdmpy.exceptions import DataNotLoadedError
from qdmpy.odmr.data import ODMRData
from qdmpy.odmr.manager import ODMR
from qdmpy.odmr.processors import NormalizationProcessor, ODMRProcessorManager


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = np.random.random((2, 2, 100, 50))  # (n_pol, n_frange, n_pixels, n_freqs)
    scan_dimensions = (10, 10)
    frequencies = np.linspace(2.87e9, 2.89e9, 50)
    return data, scan_dimensions, frequencies


@pytest.fixture
def sample_odmr_data(sample_data):
    """Create a sample ODMRData instance for testing."""
    data, scan_dimensions, frequencies = sample_data
    return ODMRData.from_numpy(data, scan_dimensions, frequencies)


class TestODMR:
    """Test class for ODMR."""

    def test_init_empty(self) -> None:
        """Test initialization with no data."""
        odmr = ODMR()
        assert odmr._raw_data is None
        assert odmr._processed_data is None
        assert odmr.is_processed is False
        assert isinstance(odmr.processor_manager, ODMRProcessorManager)

    def test_init_with_data(self, sample_odmr_data) -> None:
        """Test initialization with data."""
        odmr = ODMR(sample_odmr_data)
        assert odmr._raw_data is sample_odmr_data
        assert odmr._processed_data is None
        assert odmr.is_processed is False
        assert isinstance(odmr.processor_manager, ODMRProcessorManager)

    def test_load_data(self, sample_data) -> None:
        """Test load_data method."""
        data, scan_dimensions, frequencies = sample_data
        odmr = ODMR()

        result = odmr.load_data(data, scan_dimensions, frequencies)
        assert result is odmr

        assert isinstance(odmr._raw_data, ODMRData)
        assert isinstance(odmr._raw_data.data, xr.DataArray)
        assert odmr._raw_data.scan_dimensions == (10, 10)
        assert odmr._raw_data.shape == (2, 2, 10, 10, 50)
        assert odmr._processed_data is None
        assert odmr.is_processed is False

    def test_reset_no_data(self) -> None:
        """Test reset method with no data."""
        odmr = ODMR()
        with pytest.raises(DataNotLoadedError, match="No raw data"):
            odmr.reset()

    def test_reset(self, sample_odmr_data) -> None:
        """Test reset method."""
        odmr = ODMR(sample_odmr_data)
        odmr._processed_data = MagicMock()
        odmr.is_processed = True

        result = odmr.reset()
        assert result is odmr
        assert odmr._processed_data is None
        assert odmr.is_processed is False

    def test_process_data_no_data(self) -> None:
        """Test process_data method with no data."""
        odmr = ODMR()
        with pytest.raises(DataNotLoadedError, match="No ODMRData loaded"):
            odmr.process_data()

    def test_process_data(self, sample_odmr_data) -> None:
        """Test process_data method."""
        odmr = ODMR(sample_odmr_data)

        mock_processor_manager = MagicMock()
        mock_processed_data = MagicMock()
        mock_processor_manager.process.return_value = mock_processed_data
        odmr.processor_manager = mock_processor_manager

        result = odmr.process_data()
        assert result is odmr
        mock_processor_manager.process.assert_called_once_with(sample_odmr_data)
        assert odmr._processed_data is mock_processed_data
        assert odmr.is_processed is True

    def test_raw_data_property_no_data(self) -> None:
        """Test raw_data property with no data."""
        odmr = ODMR()
        with pytest.raises(DataNotLoadedError, match="No raw data available"):
            odmr.raw_data

    def test_raw_data_property(self, sample_odmr_data) -> None:
        """Test raw_data property."""
        odmr = ODMR(sample_odmr_data)
        assert odmr.raw_data is sample_odmr_data

    def test_processed_data_property_no_data(self) -> None:
        """Test processed_data property with no processed data."""
        odmr = ODMR()
        with pytest.raises(DataNotLoadedError, match="No processed data available"):
            odmr.processed_data

    def test_processed_data_property(self, sample_odmr_data) -> None:
        """Test processed_data property."""
        odmr = ODMR(sample_odmr_data)
        mock_processed_data = MagicMock()
        odmr._processed_data = mock_processed_data
        assert odmr.processed_data is mock_processed_data

    def test_method_chaining(self, sample_data) -> None:
        """Test that methods can be chained."""
        data, scan_dimensions, frequencies = sample_data

        odmr = ODMR()
        odmr.processor_manager.add_processor(NormalizationProcessor())

        result = odmr.load_data(data, scan_dimensions, frequencies).process_data().reset()
        assert result is odmr
