"""Test module for QDMpy.odmr.processors."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from qdmpy.odmr.data import ODMRData
from qdmpy.odmr.processors import (
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
            BaseProcessor()  # type: ignore[abstract]

    def test_to_config_round_trip(self) -> None:
        """Test that to_config produces a JSON-compatible dict."""
        processor = NormalizationProcessor()
        config = processor.to_config()
        assert config == {"type": "NormalizationProcessor", "method": "mean"}

    def test_frozen_prevents_mutation(self) -> None:
        """Test that processor fields cannot be mutated after construction."""
        processor = NormalizationProcessor()
        with pytest.raises(ValidationError):
            processor.method = "min"  # type: ignore[misc]


class TestNormalizationProcessor:
    """Test class for NormalizationProcessor."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        processor = NormalizationProcessor()
        assert processor.method == "mean"

    def test_type_field(self) -> None:
        """Test that type discriminator field is correct."""
        processor = NormalizationProcessor()
        assert processor.type == "NormalizationProcessor"

    def test_process_mean_method(self, sample_odmr_data) -> None:
        """Test that mean normalization divides each pixel by its mean intensity."""
        processor = NormalizationProcessor()
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)

        # After mean-normalisation, each pixel's mean across freq_idx should be 1.0
        mean_values = result.data.mean(dim="freq_idx")
        np.testing.assert_allclose(mean_values.values, 1.0, rtol=1e-12)

    def test_process_max_deprecated(self) -> None:
        """Test that method='max' raises DeprecationWarning (not an error)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NormalizationProcessor(method="max")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    def test_process_max_normalizes_by_max(self, sample_odmr_data) -> None:
        """Test that max normalization divides each pixel by its max intensity."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            processor = NormalizationProcessor(method="max")
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        # After max-normalization, each pixel's max across freq_idx should be 1.0
        max_values = result.data.max(dim="freq_idx")
        np.testing.assert_allclose(max_values.values, 1.0, rtol=1e-6)

    def test_process_unsupported_method_raises_validation_error(self) -> None:
        """Test that an unknown method is rejected at construction time."""
        with pytest.raises(ValidationError):
            NormalizationProcessor(method="unsupported")

    def test_to_config(self) -> None:
        """Test serialization to config dict."""
        processor = NormalizationProcessor()
        config = processor.to_config()
        assert config == {"type": "NormalizationProcessor", "method": "mean"}


class TestBinningProcessor:
    """Test class for BinningProcessor."""

    def test_init(self) -> None:
        """Test initialization with valid parameters."""
        processor = BinningProcessor(bin_factor=2)
        assert processor.bin_factor == 2

    def test_init_invalid(self) -> None:
        """Test initialization with invalid parameters raises Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            BinningProcessor(bin_factor=0)

        with pytest.raises(ValidationError):
            BinningProcessor(bin_factor=-1)

    def test_type_field(self) -> None:
        """Test that type discriminator field is correct."""
        processor = BinningProcessor(bin_factor=4)
        assert processor.type == "BinningProcessor"

    def test_process(self, sample_odmr_data) -> None:
        """Test process method reduces spatial dimensions."""
        processor = BinningProcessor(bin_factor=2)
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)

        # Original 10x10, binned by 2 -> 5x5
        assert result.data.shape == (2, 2, 5, 5, 50)

    def test_to_config(self) -> None:
        """Test serialization to config dict."""
        processor = BinningProcessor(bin_factor=4)
        config = processor.to_config()
        assert config == {"type": "BinningProcessor", "bin_factor": 4}


class TestOutlierProcessor:
    """Test class for OutlierProcessor."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        processor = OutlierProcessor()
        assert processor.z_score_threshold == 0.003

    def test_init_custom(self) -> None:
        """Test initialization with custom parameters."""
        processor = OutlierProcessor(z_score_threshold=0.01)
        assert processor.z_score_threshold == 0.01

    def test_init_invalid(self) -> None:
        """Test that non-positive threshold raises ValidationError."""
        with pytest.raises(ValidationError):
            OutlierProcessor(z_score_threshold=0.0)

        with pytest.raises(ValidationError):
            OutlierProcessor(z_score_threshold=-1.0)

    def test_type_field(self) -> None:
        """Test that type discriminator field is correct."""
        processor = OutlierProcessor()
        assert processor.type == "OutlierProcessor"

    def test_process(self, sample_odmr_data) -> None:
        """Test process method masks outlier values as NaN."""
        sample_odmr_data.data.values[0, 0, 0, 0, 0] = 1000.0

        processor = OutlierProcessor(z_score_threshold=0.1)
        result = processor.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)
        assert np.isnan(result.data.values[0, 0, 0, 0, 0])

    def test_to_config(self) -> None:
        """Test serialization to config dict."""
        processor = OutlierProcessor(z_score_threshold=0.01)
        config = processor.to_config()
        assert config == {"type": "OutlierProcessor", "z_score_threshold": 0.01}


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

    def test_init_invalid(self) -> None:
        """Test that non-positive factor raises ValidationError."""
        with pytest.raises(ValidationError):
            FluorescenceCorrectionProcessor(correction_factor=0.0)

    def test_type_field(self) -> None:
        """Test that type discriminator field is correct."""
        processor = FluorescenceCorrectionProcessor()
        assert processor.type == "FluorescenceCorrectionProcessor"

    def test_process(self, sample_odmr_data, monkeypatch) -> None:
        """Test process method applies fluorescence correction."""
        mock_baseline = xr.DataArray(
            np.ones((2, 2, 50)) * 0.1,
            dims=("polarity", "freq_range", "freq_idx"),
        )
        monkeypatch.setattr(
            "qdmpy.odmr.processors.analyze_fluorescence_effects",
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

    def test_process_uses_init_factor(self, sample_odmr_data, monkeypatch) -> None:
        """Test process method uses correction_factor set at init time."""
        mock_baseline = xr.DataArray(
            np.ones((2, 2, 50)) * 0.1,
            dims=("polarity", "freq_range", "freq_idx"),
        )
        monkeypatch.setattr(
            "qdmpy.odmr.processors.analyze_fluorescence_effects",
            lambda data, pixel_idx=None: (0, mock_baseline),
        )

        processor = FluorescenceCorrectionProcessor(correction_factor=0.5)
        original_values = sample_odmr_data.data.values.copy()
        result = processor.process(sample_odmr_data)

        expected_data = original_values - 0.05
        np.testing.assert_allclose(result.data.values, expected_data)

    def test_to_config(self) -> None:
        """Test serialization to config dict."""
        processor = FluorescenceCorrectionProcessor(correction_factor=0.3)
        config = processor.to_config()
        assert config == {"type": "FluorescenceCorrectionProcessor", "correction_factor": 0.3}


class TestFluorescenceAnalysis:
    """Test class for fluorescence analysis functions."""

    def test_analyze_fluorescence_effects(self, sample_odmr_data) -> None:
        """Test analyze_fluorescence_effects with a specified pixel."""
        sample_odmr_data.data.values[:] = 1.0
        sample_odmr_data.data.values[:, :, 5, 0, :] = 0.8

        idx, baseline_corrected = analyze_fluorescence_effects(
            sample_odmr_data,
            pixel_idx=50,
        )

        assert idx == 50

        assert isinstance(baseline_corrected, xr.DataArray)
        assert baseline_corrected.dims == ("polarity", "freq_range", "freq_idx")

        assert -0.5 < float(baseline_corrected.mean()) < 0.5

    def test_analyze_fluorescence_effects_auto_pixel(self, sample_odmr_data) -> None:
        """Test analyze_fluorescence_effects with automatic pixel selection."""
        idx, baseline_corrected = analyze_fluorescence_effects(sample_odmr_data)

        assert isinstance(idx, int)
        n_pixels = sample_odmr_data.data.sizes["y"] * sample_odmr_data.data.sizes["x"]
        assert 0 <= idx < n_pixels

        assert isinstance(baseline_corrected, xr.DataArray)
        assert baseline_corrected.dims == ("polarity", "freq_range", "freq_idx")


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

    def test_process_chains_sequentially(self, sample_odmr_data) -> None:
        """Test process method chains processors and writes pipeline metadata."""
        manager = ODMRProcessorManager()
        manager.add_processor(NormalizationProcessor())
        manager.add_processor(BinningProcessor(bin_factor=2))

        result = manager.process(sample_odmr_data)

        assert result is not sample_odmr_data
        assert isinstance(result, ODMRData)
        # Normalization then binning: 10x10 -> 5x5
        assert result.data.sizes["y"] == 5
        assert result.data.sizes["x"] == 5

    def test_process_writes_pipeline_metadata(self, sample_odmr_data) -> None:
        """Test that process() writes a complete pipeline snapshot to metadata."""
        manager = ODMRProcessorManager()
        manager.add_processor(NormalizationProcessor())
        manager.add_processor(BinningProcessor(bin_factor=4))

        result = manager.process(sample_odmr_data)

        assert "pipeline" in result.metadata
        pipeline = result.metadata["pipeline"]
        assert len(pipeline) == 2
        assert pipeline[0] == {"type": "NormalizationProcessor", "method": "mean"}
        assert pipeline[1] == {"type": "BinningProcessor", "bin_factor": 4}

    def test_process_empty_pipeline(self, sample_odmr_data) -> None:
        """Test that an empty pipeline writes an empty pipeline list to metadata."""
        manager = ODMRProcessorManager()
        result = manager.process(sample_odmr_data)

        assert "pipeline" in result.metadata
        assert result.metadata["pipeline"] == []

    def test_list_processors(self) -> None:
        """Test list_processors returns type names."""
        manager = ODMRProcessorManager()

        assert manager.list_processors() == []

        manager.add_processor(NormalizationProcessor())
        manager.add_processor(BinningProcessor(bin_factor=2))

        processor_names = manager.list_processors()
        assert len(processor_names) == 2
        assert processor_names[0] == "NormalizationProcessor"
        assert processor_names[1] == "BinningProcessor"

    def test_pipeline_config_property(self) -> None:
        """Test pipeline_config property returns serializable list."""
        manager = ODMRProcessorManager()
        manager.add_processor(NormalizationProcessor())
        manager.add_processor(OutlierProcessor(z_score_threshold=0.01))

        config = manager.pipeline_config
        assert config == [
            {"type": "NormalizationProcessor", "method": "mean"},
            {"type": "OutlierProcessor", "z_score_threshold": 0.01},
        ]

    def test_from_config_round_trip(self) -> None:
        """Test that from_config reconstructs a pipeline identical to the original."""
        original = ODMRProcessorManager()
        original.add_processor(NormalizationProcessor())
        original.add_processor(BinningProcessor(bin_factor=2))
        original.add_processor(OutlierProcessor(z_score_threshold=0.005))

        config = original.pipeline_config
        restored = ODMRProcessorManager.from_config(config)

        assert restored.pipeline_config == config
        assert len(restored.processors) == 3
        assert isinstance(restored.processors[0], NormalizationProcessor)
        assert isinstance(restored.processors[1], BinningProcessor)
        assert restored.processors[1].bin_factor == 2
        assert isinstance(restored.processors[2], OutlierProcessor)
        assert restored.processors[2].z_score_threshold == 0.005

    def test_from_config_metadata_round_trip(self, sample_odmr_data) -> None:
        """Test reconstructing a pipeline from processed_data metadata."""
        manager = ODMRProcessorManager()
        manager.add_processor(BinningProcessor(bin_factor=2))

        processed = manager.process(sample_odmr_data)
        pipeline_config = processed.metadata["pipeline"]

        restored = ODMRProcessorManager.from_config(pipeline_config)
        assert restored.pipeline_config == pipeline_config
