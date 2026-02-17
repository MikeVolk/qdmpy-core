"""Test module for QDMpy.odmr.processors."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from QDMpy.exceptions import DataValidationError
from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.processors import (
    BaseProcessor,
    BinningProcessor,
    FluorescenceCorrectionProcessor,
    NormalizationProcessor,
    ODMRProcessorManager,
    OutlierProcessor,
    analyze_fluorescence_effects,
)


@pytest.fixture
def sample_odmr_data():
    """Create a real ODMRData instance for testing."""
    rng = np.random.default_rng(42)
    data = rng.random((2, 2, 100, 50))
    scan_dimensions = (10, 10)
    frequencies = np.linspace(2.87e9, 2.89e9, 50)
    return ODMRData.from_numpy(data, scan_dimensions, frequencies)


class TestBaseProcessor:
    """Test class for BaseProcessor."""

    def test_abstract_class(self) -> None:
        """Test that BaseProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProcessor()


class TestNormalizationProcessor:
    """Test class for NormalizationProcessor."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        processor = NormalizationProcessor()
        assert processor.method == 'max'

    def test_init_custom(self) -> None:
        """Test initialization with custom parameters."""
        processor = NormalizationProcessor(method='custom')
        assert processor.method == 'custom'

    def test_process_max_method(self, sample_odmr_data) -> None:
        """Test process method with 'max' normalization."""
        processor = NormalizationProcessor(method='max')
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)

        max_values = result.data.max(dim='freq_idx')
        np.testing.assert_allclose(max_values.values, 1.0)

        assert result.metadata['normalized'] is True

    def test_process_unsupported_method(self, sample_odmr_data) -> None:
        """Test process method with unsupported normalization method."""
        processor = NormalizationProcessor(method='unsupported')
        with pytest.raises(NotImplementedError):
            processor.process(sample_odmr_data)


class TestBinningProcessor:
    """Test class for BinningProcessor."""

    def test_init(self) -> None:
        """Test initialization with valid parameters."""
        processor = BinningProcessor(bin_factor=2)
        assert processor.bin_factor == 2

    def test_init_invalid(self) -> None:
        """Test initialization with invalid parameters."""
        with pytest.raises(DataValidationError):
            BinningProcessor(bin_factor=0)

        with pytest.raises(DataValidationError):
            BinningProcessor(bin_factor=-1)

    def test_process(self, sample_odmr_data) -> None:
        """Test process method reduces spatial dimensions."""
        processor = BinningProcessor(bin_factor=2)
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)

        # Original 10x10, binned by 2 -> 5x5
        assert result.data.shape == (2, 2, 5, 5, 50)

        assert result.metadata['binned'] is True
        assert result.metadata['bin_factor'] == 2


class TestOutlierProcessor:
    """Test class for OutlierProcessor."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        processor = OutlierProcessor()
        assert processor.threshold == 0.001

    def test_init_custom(self) -> None:
        """Test initialization with custom parameters."""
        processor = OutlierProcessor(threshold=0.01)
        assert processor.threshold == 0.01

    def test_process(self, sample_odmr_data) -> None:
        """Test process method masks outlier values as NaN."""
        sample_odmr_data.data.values[0, 0, 0, 0, 0] = 1000.0

        processor = OutlierProcessor(threshold=0.1)
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)
        assert np.isnan(result.data.values[0, 0, 0, 0, 0])

        assert 'outlier_masking' in result.metadata
        assert result.metadata['outlier_masking']['threshold'] == 0.1


class TestFluorescenceCorrectionProcessor:
    """Test class for FluorescenceCorrectionProcessor."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        processor = FluorescenceCorrectionProcessor()
        assert processor.correction_factor == 0.2

    def test_init_custom(self) -> None:
        """Test initialization with custom parameters."""
        processor = FluorescenceCorrectionProcessor(correction_factor=0.5)
        assert processor.correction_factor == 0.5

    def test_process(self, sample_odmr_data, monkeypatch) -> None:
        """Test process method applies fluorescence correction."""
        mock_baseline = xr.DataArray(
            np.ones((2, 2, 50)) * 0.1,
            dims=('polarity', 'freq_range', 'freq_idx'),
        )
        monkeypatch.setattr(
            'QDMpy.odmr.processors.analyze_fluorescence_effects',
            lambda data, pixel_idx=None: (0, mock_baseline),
        )

        processor = FluorescenceCorrectionProcessor()
        original_values = sample_odmr_data.data.values.copy()
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)

        # Expected: factor (0.2) * baseline_corrected (0.1) = 0.02
        expected_data = original_values - 0.02
        np.testing.assert_allclose(result.data.values, expected_data)

        assert 'fluorescence_correction' in result.metadata
        assert result.metadata['fluorescence_correction']['factor'] == 0.2
        assert result.metadata['fluorescence_correction']['applied'] is True

    def test_process_with_override_factor(self, sample_odmr_data, monkeypatch) -> None:
        """Test process method with override correction factor."""
        mock_baseline = xr.DataArray(
            np.ones((2, 2, 50)) * 0.1,
            dims=('polarity', 'freq_range', 'freq_idx'),
        )
        monkeypatch.setattr(
            'QDMpy.odmr.processors.analyze_fluorescence_effects',
            lambda data, pixel_idx=None: (0, mock_baseline),
        )

        processor = FluorescenceCorrectionProcessor(correction_factor=0.2)
        original_values = sample_odmr_data.data.values.copy()
        result = processor.process(sample_odmr_data, correction_factor=0.5)

        expected_data = original_values - 0.05
        np.testing.assert_allclose(result.data.values, expected_data)

        assert result.metadata['fluorescence_correction']['factor'] == 0.5

    def test_process_with_legacy_param(self, sample_odmr_data, monkeypatch) -> None:
        """Test process method with legacy glob_fluorescence parameter."""
        mock_baseline = xr.DataArray(
            np.ones((2, 2, 50)) * 0.1,
            dims=('polarity', 'freq_range', 'freq_idx'),
        )
        monkeypatch.setattr(
            'QDMpy.odmr.processors.analyze_fluorescence_effects',
            lambda data, pixel_idx=None: (0, mock_baseline),
        )

        processor = FluorescenceCorrectionProcessor(correction_factor=0.2)
        original_values = sample_odmr_data.data.values.copy()
        result = processor.process(sample_odmr_data, glob_fluorescence=0.3)

        expected_data = original_values - 0.03
        np.testing.assert_allclose(result.data.values, expected_data)

        assert result.metadata['fluorescence_correction']['factor'] == 0.3


class TestFluorescenceAnalysis:
    """Test class for fluorescence analysis functions."""

    def test_analyze_fluorescence_effects(self, sample_odmr_data) -> None:
        """Test analyze_fluorescence_effects with a specified pixel."""
        sample_odmr_data.data.values[:] = 1.0
        sample_odmr_data.data.values[:, :, 5, 0, :] = 0.8

        idx, baseline_corrected = analyze_fluorescence_effects(
            sample_odmr_data, pixel_idx=50,
        )

        assert idx == 50

        assert isinstance(baseline_corrected, xr.DataArray)
        assert baseline_corrected.dims == ('polarity', 'freq_range', 'freq_idx')

        assert -0.5 < float(baseline_corrected.mean()) < 0.5

    def test_analyze_fluorescence_effects_auto_pixel(self, sample_odmr_data) -> None:
        """Test analyze_fluorescence_effects with automatic pixel selection."""
        idx, baseline_corrected = analyze_fluorescence_effects(sample_odmr_data)

        assert isinstance(idx, int)
        n_pixels = sample_odmr_data.data.sizes['y'] * sample_odmr_data.data.sizes['x']
        assert 0 <= idx < n_pixels

        assert isinstance(baseline_corrected, xr.DataArray)
        assert baseline_corrected.dims == ('polarity', 'freq_range', 'freq_idx')


class TestODMRProcessorManager:
    """Test class for ODMRProcessorManager."""

    def test_init(self) -> None:
        """Test initialization."""
        manager = ODMRProcessorManager()
        assert len(manager.processors) == 0

    def test_add_processor(self) -> None:
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

    def test_process(self, sample_odmr_data) -> None:
        """Test process method chains processors sequentially."""
        manager = ODMRProcessorManager()

        processor1 = MagicMock()
        processor1.__class__.__name__ = 'MockProcessor1'
        processor1.process.return_value = sample_odmr_data

        processor2 = MagicMock()
        processor2.__class__.__name__ = 'MockProcessor2'
        processor2.process.return_value = sample_odmr_data

        manager.add_processor(processor1)
        manager.add_processor(processor2)

        result = manager.process(sample_odmr_data)

        processor1.process.assert_called_once_with(sample_odmr_data)
        processor2.process.assert_called_once_with(sample_odmr_data)

        assert result is sample_odmr_data

    def test_list_processors(self) -> None:
        """Test list_processors method."""
        manager = ODMRProcessorManager()

        assert manager.list_processors() == []

        manager.add_processor(NormalizationProcessor())
        manager.add_processor(BinningProcessor(bin_factor=2))

        processor_names = manager.list_processors()
        assert len(processor_names) == 2
        assert processor_names[0] == 'NormalizationProcessor'
        assert processor_names[1] == 'BinningProcessor'
