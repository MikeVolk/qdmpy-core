"""
Test module for QDMpy.odmr.processors
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from QDMpy.odmr.processors import (
    BaseProcessor,
    NormalizationProcessor,
    BinningProcessor,
    OutlierProcessor,
    ODMRProcessorManager
)


@pytest.fixture
def sample_odmr_data():
    """Create a mock ODMRData instance for testing."""
    mock_data = MagicMock()
    # Sample data with shape (2, 3, 100, 50)
    # (modes, reps, pixels, frequencies)
    mock_data.data = np.random.random((2, 3, 100, 50))
    mock_data.scan_dimensions = np.array([10, 10])  # 10x10 grid
    mock_data.frequencies = np.linspace(2.87e9, 2.89e9, 50)
    mock_data.metadata = {}
    mock_data.__class__ = MagicMock(return_value=mock_data)
    return mock_data


class TestBaseProcessor:
    """Test class for BaseProcessor."""

    def test_abstract_class(self):
        """Test that BaseProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProcessor()


class TestNormalizationProcessor:
    """Test class for NormalizationProcessor."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        processor = NormalizationProcessor()
        assert processor.method == "max"

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        processor = NormalizationProcessor(method="custom")
        assert processor.method == "custom"

    def test_process_max_method(self, sample_odmr_data):
        """Test process method with 'max' normalization."""
        processor = NormalizationProcessor(method="max")
        result = processor.process(sample_odmr_data)
        
        # Check the result is a new instance
        assert result is not sample_odmr_data
        
        # Get the max values along the frequency axis
        max_values = np.max(sample_odmr_data.data, axis=-1, keepdims=True)
        expected = sample_odmr_data.data / max_values
        
        # Check the normalized data matches expected
        np.testing.assert_allclose(result.data, expected)
        
        # Check metadata is updated
        assert result.metadata['normalized'] == True

    def test_process_unsupported_method(self, sample_odmr_data):
        """Test process method with unsupported normalization method."""
        processor = NormalizationProcessor(method="unsupported")
        with pytest.raises(NotImplementedError):
            processor.process(sample_odmr_data)


class TestBinningProcessor:
    """Test class for BinningProcessor."""

    def test_init(self):
        """Test initialization with valid parameters."""
        processor = BinningProcessor(bin_factor=2)
        assert processor.bin_factor == 2

    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            BinningProcessor(bin_factor=0)
        
        with pytest.raises(ValueError):
            BinningProcessor(bin_factor=-1)

    def test_process(self, sample_odmr_data):
        """Test process method."""
        processor = BinningProcessor(bin_factor=2)
        result = processor.process(sample_odmr_data)
        
        # Check the result is a new instance
        assert result is not sample_odmr_data
        
        # The spatial dimensions should be reduced
        # Original is (2, 3, 100, 50)
        # Original spatial shape is 10x10 (= 100 pixels)
        # After binning by 2, should be 5x5 (= 25 pixels)
        assert result.data.shape[2] < sample_odmr_data.data.shape[2]
        
        # Check metadata is updated
        assert result.metadata['binned'] == True
        assert result.metadata['bin_factor'] == 2


class TestOutlierProcessor:
    """Test class for OutlierProcessor."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        processor = OutlierProcessor()
        assert processor.threshold == 0.001

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        processor = OutlierProcessor(threshold=0.01)
        assert processor.threshold == 0.01

    def test_process(self, sample_odmr_data):
        """Test process method."""
        # Add some outliers to the data
        data_with_outliers = sample_odmr_data.data.copy()
        data_with_outliers[0, 0, 0, 0] = 1000  # Extreme value
        sample_odmr_data.data = data_with_outliers
        
        processor = OutlierProcessor(threshold=0.1)
        result = processor.process(sample_odmr_data)
        
        # Check the result is a new instance
        assert result is not sample_odmr_data
        
        # Verify our outlier is masked (should be NaN)
        assert np.isnan(result.data[0, 0, 0, 0])
        
        # Check metadata is updated
        assert 'outlier_masking' in result.metadata
        assert result.metadata['outlier_masking']['threshold'] == 0.1


class TestODMRProcessorManager:
    """Test class for ODMRProcessorManager."""

    def test_init(self):
        """Test initialization."""
        manager = ODMRProcessorManager()
        assert len(manager.processors) == 0

    def test_add_processor(self):
        """Test add_processor method."""
        manager = ODMRProcessorManager()
        processor1 = NormalizationProcessor()
        processor2 = BinningProcessor(bin_factor=2)
        
        manager.add_processor(processor1)
        assert len(manager.processors) == 1
        
        manager.add_processor(processor2)
        assert len(manager.processors) == 2
        
        assert manager.processors[0] is processor1
        assert manager.processors[1] is processor2

    def test_process(self, sample_odmr_data):
        """Test process method."""
        manager = ODMRProcessorManager()
        
        # Add mock processors that each append to metadata
        processor1 = MagicMock()
        processor1.__class__.__name__ = "MockProcessor1"
        processor1.process.return_value = sample_odmr_data
        
        processor2 = MagicMock()
        processor2.__class__.__name__ = "MockProcessor2"
        processor2.process.return_value = sample_odmr_data
        
        manager.add_processor(processor1)
        manager.add_processor(processor2)
        
        result = manager.process(sample_odmr_data)
        
        # Verify that each processor was called once with the sample data
        processor1.process.assert_called_once_with(sample_odmr_data)
        processor2.process.assert_called_once_with(sample_odmr_data)
        
        # The result should be the final processed data
        assert result is sample_odmr_data

    def test_list_processors(self):
        """Test list_processors method."""
        manager = ODMRProcessorManager()
        
        # Empty manager should return empty list
        assert manager.list_processors() == []
        
        # Add processors and check the list
        manager.add_processor(NormalizationProcessor())
        manager.add_processor(BinningProcessor(bin_factor=2))
        
        processor_names = manager.list_processors()
        assert len(processor_names) == 2
        assert processor_names[0] == "NormalizationProcessor"
        assert processor_names[1] == "BinningProcessor"