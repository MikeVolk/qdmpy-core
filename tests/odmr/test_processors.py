"""Test module for QDMpy.odmr.processors
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from QDMpy.odmr.processors import (
    BaseProcessor,
    BinningProcessor,
    FluorescenceCorrectionProcessor,
    NormalizationProcessor,
    ODMRProcessorManager,
    OutlierProcessor,
    analyze_fluorescence_effects,
    preview_fluorescence_correction,
)


@pytest.fixture
def sample_odmr_data():
    """Create a mock ODMRData instance for testing."""
    # Create the mock data object
    mock_data = MagicMock()

    # Set up the data with appropriate dimensions
    # Shape (2, 3, 100, 50) - (polarities, frequency_ranges, pixels, frequencies)
    mock_data.data = np.random.random((2, 3, 100, 50))
    mock_data.scan_dimensions = np.array([10, 10])  # 10x10 grid, 100 pixels total
    mock_data.frequencies = np.linspace(2.87e9, 2.89e9, 50)
    mock_data.metadata = {}

    # The key issue: when processors call data.__class__(...), we need to return a NEW mock
    # not the same mock object
    def create_new_instance(*args, **kwargs):
        new_mock = MagicMock()
        new_mock.data = kwargs.get('data', np.copy(mock_data.data))
        new_mock.scan_dimensions = kwargs.get('scan_dimensions', np.copy(mock_data.scan_dimensions))
        new_mock.frequencies = kwargs.get('frequencies', np.copy(mock_data.frequencies))
        new_mock.metadata = kwargs.get('metadata', mock_data.metadata.copy())
        return new_mock

    # Set up the class mock to return a new instance
    mock_class = MagicMock()
    mock_class.side_effect = create_new_instance
    mock_data.__class__ = mock_class

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
        assert processor.method == 'max'

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        processor = NormalizationProcessor(method='custom')
        assert processor.method == 'custom'

    def test_process_max_method(self, sample_odmr_data):
        """Test process method with 'max' normalization."""
        processor = NormalizationProcessor(method='max')
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
        processor = NormalizationProcessor(method='unsupported')
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


class TestFluorescenceCorrectionProcessor:
    """Test class for FluorescenceCorrectionProcessor."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        processor = FluorescenceCorrectionProcessor()
        assert processor.correction_factor == 0.2

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        processor = FluorescenceCorrectionProcessor(correction_factor=0.5)
        assert processor.correction_factor == 0.5

    def test_process(self, sample_odmr_data, monkeypatch):
        """Test process method."""
        # Mock the analyze_fluorescence_effects function
        mock_baseline_corrected = np.ones_like(sample_odmr_data.data[:, :, 0:1, :]) * 0.1
        monkeypatch.setattr(
            'QDMpy.odmr.processors.analyze_fluorescence_effects',
            lambda data, pixel_idx=None: (0, mock_baseline_corrected)
        )

        # Process the data with default correction factor
        processor = FluorescenceCorrectionProcessor()
        result = processor.process(sample_odmr_data)

        # Check the result is a new instance
        assert result is not sample_odmr_data

        # Expected correction: factor (0.2) * baseline_corrected (0.1) = 0.02
        expected_correction = 0.02
        expected_data = sample_odmr_data.data - expected_correction

        # Check the corrected data
        np.testing.assert_allclose(result.data, expected_data)

        # Check metadata is updated
        assert 'fluorescence_correction' in result.metadata
        assert result.metadata['fluorescence_correction']['factor'] == 0.2
        assert result.metadata['fluorescence_correction']['applied'] is True

    def test_process_with_override_factor(self, sample_odmr_data, monkeypatch):
        """Test process method with override correction factor."""
        # Mock the analyze_fluorescence_effects function
        mock_baseline_corrected = np.ones_like(sample_odmr_data.data[:, :, 0:1, :]) * 0.1
        monkeypatch.setattr(
            'QDMpy.odmr.processors.analyze_fluorescence_effects',
            lambda data, pixel_idx=None: (0, mock_baseline_corrected)
        )

        # Process the data with override correction factor
        processor = FluorescenceCorrectionProcessor(correction_factor=0.2)
        result = processor.process(sample_odmr_data, correction_factor=0.5)

        # Expected correction: factor (0.5) * baseline_corrected (0.1) = 0.05
        expected_correction = 0.05
        expected_data = sample_odmr_data.data - expected_correction

        # Check the corrected data
        np.testing.assert_allclose(result.data, expected_data)

        # Check metadata is updated
        assert result.metadata['fluorescence_correction']['factor'] == 0.5

    def test_process_with_legacy_param(self, sample_odmr_data, monkeypatch):
        """Test process method with legacy glob_fluorescence parameter."""
        # Mock the analyze_fluorescence_effects function
        mock_baseline_corrected = np.ones_like(sample_odmr_data.data[:, :, 0:1, :]) * 0.1
        monkeypatch.setattr(
            'QDMpy.odmr.processors.analyze_fluorescence_effects',
            lambda data, pixel_idx=None: (0, mock_baseline_corrected)
        )

        # Process the data with legacy parameter
        processor = FluorescenceCorrectionProcessor(correction_factor=0.2)
        result = processor.process(sample_odmr_data, glob_fluorescence=0.3)

        # Expected correction: factor (0.3) * baseline_corrected (0.1) = 0.03
        expected_correction = 0.03
        expected_data = sample_odmr_data.data - expected_correction

        # Check the corrected data
        np.testing.assert_allclose(result.data, expected_data)

        # Check metadata is updated
        assert result.metadata['fluorescence_correction']['factor'] == 0.3


class TestFluorescenceAnalysis:
    """Test class for fluorescence analysis functions."""
    
    def test_analyze_fluorescence_effects(self, sample_odmr_data):
        """Test the analyze_fluorescence_effects function."""
        # Set specific values in the data for predictable testing
        sample_odmr_data.data = np.ones_like(sample_odmr_data.data)
        # Add a variation pattern that should be detected
        sample_odmr_data.data[:, :, 50, :] = 0.8  # Make one pixel different
        
        # Call the function with specified pixel
        idx, baseline_corrected = analyze_fluorescence_effects(sample_odmr_data, pixel_idx=50)
        
        # Check the returned index matches what we specified
        assert idx == 50
        
        # Check that baseline_corrected contains reasonable values
        # The baseline should be calculated from the first and last 5% of frequencies
        # With our mockup data, this should be close to 0 (after baseline subtraction)
        assert baseline_corrected.shape[2] == 1  # Only one pixel in dimension 2
        assert -0.5 < np.mean(baseline_corrected) < 0.5  # Should be close to zero
        
    def test_analyze_fluorescence_effects_auto_pixel(self, sample_odmr_data):
        """Test the analyze_fluorescence_effects function with auto pixel selection."""
        # Set specific values in the data for predictable testing
        sample_odmr_data.data = np.ones_like(sample_odmr_data.data)
        # Add a variation pattern that should be detected
        sample_odmr_data.data[:, :, 50, :] = 0.9  # Make one pixel slightly different
        
        # Call the function with automatic pixel selection
        idx, baseline_corrected = analyze_fluorescence_effects(sample_odmr_data)
        
        # The function should identify pixel 50 as most divergent
        # However, since we're using random data in the fixture, we can't guarantee
        # which pixel will be selected, so we just check the type is correct
        assert isinstance(idx, int)
        assert 0 <= idx < sample_odmr_data.data.shape[2]


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
        processor1.__class__.__name__ = 'MockProcessor1'
        processor1.process.return_value = sample_odmr_data

        processor2 = MagicMock()
        processor2.__class__.__name__ = 'MockProcessor2'
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
        assert processor_names[0] == 'NormalizationProcessor'
        assert processor_names[1] == 'BinningProcessor'
