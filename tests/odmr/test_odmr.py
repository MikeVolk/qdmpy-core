"""
Test module for QDMpy.odmr.odmr
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from QDMpy.odmr.odmr import ODMR
from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.processors import ODMRProcessorManager, BinningProcessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = np.random.random((2, 3, 100, 50))  # (modes, reps, pixels, frequencies)
    scan_dimensions = np.array([10, 10])  # 10x10 grid
    frequencies = np.linspace(2.87e9, 2.89e9, 50)  # 50 frequencies
    return data, scan_dimensions, frequencies


@pytest.fixture
def sample_odmr_data(sample_data):
    """Create a sample ODMRData instance for testing."""
    data, scan_dimensions, frequencies = sample_data
    return ODMRData(data, scan_dimensions, frequencies)


class TestODMR:
    """Test class for ODMR."""

    def test_init_empty(self):
        """Test initialization with no data."""
        odmr = ODMR()
        assert odmr._raw_data is None
        assert odmr._processed_data is None
        assert odmr.is_processed is False
        assert isinstance(odmr.processor_manager, ODMRProcessorManager)

    def test_init_with_data(self, sample_odmr_data):
        """Test initialization with data."""
        odmr = ODMR(sample_odmr_data)
        assert odmr._raw_data is sample_odmr_data
        assert odmr._processed_data is None
        assert odmr.is_processed is False
        assert isinstance(odmr.processor_manager, ODMRProcessorManager)

    def test_load_data(self, sample_data):
        """Test load_data method."""
        data, scan_dimensions, frequencies = sample_data
        odmr = ODMR()
        
        # Test method returns self for chaining
        result = odmr.load_data(data, scan_dimensions, frequencies)
        assert result is odmr
        
        # Test data is loaded correctly
        assert isinstance(odmr._raw_data, ODMRData)
        assert odmr._raw_data.data is data
        assert odmr._raw_data.scan_dimensions is scan_dimensions
        assert odmr._raw_data.frequencies is frequencies
        assert odmr._processed_data is None
        assert odmr.is_processed is False

    def test_reset_no_data(self):
        """Test reset method with no data."""
        odmr = ODMR()
        with pytest.raises(ValueError, match="No raw data"):
            odmr.reset()

    def test_reset(self, sample_odmr_data):
        """Test reset method."""
        odmr = ODMR(sample_odmr_data)
        
        # Create a processed data state
        odmr._processed_data = MagicMock()
        odmr.is_processed = True
        
        # Test method returns self for chaining
        result = odmr.reset()
        assert result is odmr
        
        # Test reset properly clears processed data
        assert odmr._processed_data is None
        assert odmr.is_processed is False

    def test_process_data_no_data(self):
        """Test process_data method with no data."""
        odmr = ODMR()
        with pytest.raises(ValueError, match="No ODMRData loaded"):
            odmr.process_data()

    def test_process_data(self, sample_odmr_data):
        """Test process_data method."""
        odmr = ODMR(sample_odmr_data)
        
        # Mock the processor_manager
        mock_processor_manager = MagicMock()
        mock_processed_data = MagicMock()
        mock_processor_manager.process.return_value = mock_processed_data
        odmr.processor_manager = mock_processor_manager
        
        # Test method returns self for chaining
        result = odmr.process_data()
        assert result is odmr
        
        # Test processing is applied correctly
        mock_processor_manager.process.assert_called_once_with(sample_odmr_data)
        assert odmr._processed_data is mock_processed_data
        assert odmr.is_processed is True

    def test_raw_data_property_no_data(self):
        """Test raw_data property with no data."""
        odmr = ODMR()
        with pytest.raises(ValueError, match="No raw data available"):
            odmr.raw_data

    def test_raw_data_property(self, sample_odmr_data):
        """Test raw_data property."""
        odmr = ODMR(sample_odmr_data)
        assert odmr.raw_data is sample_odmr_data

    def test_processed_data_property_no_data(self):
        """Test processed_data property with no processed data."""
        odmr = ODMR()
        with pytest.raises(ValueError, match="No processed data available"):
            odmr.processed_data

    def test_processed_data_property(self, sample_odmr_data):
        """Test processed_data property."""
        odmr = ODMR(sample_odmr_data)
        mock_processed_data = MagicMock()
        odmr._processed_data = mock_processed_data
        assert odmr.processed_data is mock_processed_data

    def test_method_chaining(self, sample_data):
        """Test that methods can be chained."""
        data, scan_dimensions, frequencies = sample_data
        
        # Create mock processor
        mock_processor = MagicMock()
        
        # Set up a chain of method calls
        odmr = ODMR()
        odmr.processor_manager.add_processor(mock_processor)
        
        result = (
            odmr.load_data(data, scan_dimensions, frequencies)
            .process_data()
            .reset()
        )
        
        # The chain should end with the ODMR instance
        assert result is odmr