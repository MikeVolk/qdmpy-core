"""
Pytest validation tests for data processing functionality.

These tests validate that the new QDMpy codebase produces identical processing
results compared to reference data generated from the old codebase.
"""

import numpy as np
import pytest
from loguru import logger


@pytest.mark.validation
@pytest.mark.processing
class TestProcessingValidation:
    """Validation tests for ODMR data processing operations."""

    @pytest.mark.parametrize("bin_factor", [1, 2, 8])
    def test_normalization_processing(
        self, reference_data, new_qdmpy_modules, test_data_folder, bin_factor
    ):
        """Test that normalization produces identical results to reference."""
        # Import new codebase modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        from QDMpy.odmr.processors import NormalizationProcessor

        # Load data with new codebase
        loader = MatlabLoader(data_folder=str(test_data_folder))
        odmr_data = ODMRData.from_loader(loader=loader)
        odmr = ODMR_new(odmr_data)

        # Apply normalization
        odmr.processor_manager.processors = []
        odmr.processor_manager.add_processor(NormalizationProcessor(method="max"))
        odmr.process_data()

        # Get reference normalized data
        ref_normalized = reference_data["normalized_data"]

        # Compare normalized data
        np.testing.assert_allclose(
            odmr.processed_data.data,
            ref_normalized,
            rtol=1e-12,
            atol=1e-15,
            err_msg="Normalized data does not match reference",
        )

        logger.info(f"✅ Normalization validation passed for bin_factor={bin_factor}")

    @pytest.mark.parametrize("bin_factor", [1, 2, 8])
    def test_binning_processing(
        self, reference_data, new_qdmpy_modules, test_data_folder, bin_factor
    ):
        """Test that binning produces identical results to reference."""
        if bin_factor == 1:
            pytest.skip("No binning applied for bin_factor=1")

        # Import new codebase modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        from QDMpy.odmr.processors import NormalizationProcessor, BinningProcessor

        # Load data with new codebase
        loader = MatlabLoader(data_folder=str(test_data_folder))
        odmr_data = ODMRData.from_loader(loader=loader)
        odmr = ODMR_new(odmr_data)

        # Apply normalization and binning
        odmr.processor_manager.processors = []
        odmr.processor_manager.add_processor(NormalizationProcessor(method="max"))
        odmr.processor_manager.add_processor(BinningProcessor(bin_factor=bin_factor))
        odmr.process_data()

        # Get reference binned data
        ref_binned = reference_data["binned_data"]
        ref_scan_dims = reference_data["binned_scan_dimensions"]

        # Compare binned data
        np.testing.assert_allclose(
            odmr.processed_data.data,
            ref_binned,
            rtol=1e-12,
            atol=1e-15,
            err_msg="Binned data does not match reference",
        )

        # Compare scan dimensions
        np.testing.assert_array_equal(
            odmr.processed_data.scan_dimensions,
            ref_scan_dims,
            err_msg="Binned scan dimensions do not match reference",
        )

        logger.info(f"✅ Binning validation passed for bin_factor={bin_factor}")
        logger.info(f"   Final shape: {odmr.processed_data.data.shape}")
        logger.info(f"   Scan dimensions: {odmr.processed_data.scan_dimensions}")

    @pytest.mark.parametrize("bin_factor", [1, 2, 8])
    def test_fluorescence_correction(
        self, reference_data, new_qdmpy_modules, test_data_folder, bin_factor
    ):
        """Test that fluorescence correction produces identical results to reference."""
        # Import new codebase modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        from QDMpy.odmr.processors import (
            NormalizationProcessor,
            BinningProcessor,
            FluorescenceCorrectionProcessor,
        )

        # Load data with new codebase
        loader = MatlabLoader(data_folder=str(test_data_folder))
        odmr_data = ODMRData.from_loader(loader=loader)
        odmr = ODMR_new(odmr_data)

        # Apply full processing pipeline
        odmr.processor_manager.processors = []
        odmr.processor_manager.add_processor(NormalizationProcessor(method="max"))
        if bin_factor > 1:
            odmr.processor_manager.add_processor(BinningProcessor(bin_factor=bin_factor))
        odmr.processor_manager.add_processor(FluorescenceCorrectionProcessor(factor=0.2))
        odmr.process_data()

        # Get reference processed data
        ref_corrected = reference_data["fluorescence_corrected_data"]

        # Compare fluorescence corrected data
        np.testing.assert_allclose(
            odmr.processed_data.data,
            ref_corrected,
            rtol=1e-12,
            atol=1e-15,
            err_msg="Fluorescence corrected data does not match reference",
        )

        logger.info(f"✅ Fluorescence correction validation passed for bin_factor={bin_factor}")


@pytest.mark.slow
@pytest.mark.validation
@pytest.mark.processing
class TestProcessingPerformance:
    """Performance validation tests for processing operations."""

    @pytest.mark.performance
    @pytest.mark.parametrize("bin_factor", [1, 2, 8])
    def test_processing_performance(
        self, reference_data, new_qdmpy_modules, test_data_folder, bin_factor, benchmark
    ):
        """Benchmark processing performance against reference times."""
        # Import new codebase modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        from QDMpy.odmr.processors import (
            NormalizationProcessor,
            BinningProcessor,
            FluorescenceCorrectionProcessor,
        )

        def run_processing():
            # Load data with new codebase
            loader = MatlabLoader(data_folder=str(test_data_folder))
            odmr_data = ODMRData.from_loader(loader=loader)
            odmr = ODMR_new(odmr_data)

            # Apply full processing pipeline
            odmr.processor_manager.processors = []
            odmr.processor_manager.add_processor(NormalizationProcessor(method="max"))
            if bin_factor > 1:
                odmr.processor_manager.add_processor(BinningProcessor(bin_factor=bin_factor))
            odmr.processor_manager.add_processor(FluorescenceCorrectionProcessor(factor=0.2))
            odmr.process_data()

            return odmr.processed_data

        # Benchmark the processing
        result = benchmark(run_processing)

        # Get reference timing if available
        ref_timing = reference_data.get("processing_time_seconds", None)
        if ref_timing is not None:
            # Allow up to 50% performance degradation
            max_allowed_time = ref_timing * 1.5
            actual_time = benchmark.stats["mean"]

            assert actual_time <= max_allowed_time, (
                f"Processing too slow: {actual_time:.3f}s vs reference {ref_timing:.3f}s "
                f"(max allowed: {max_allowed_time:.3f}s)"
            )

            logger.info(f"✅ Performance validation passed for bin_factor={bin_factor}")
            logger.info(f"   New: {actual_time:.3f}s vs Reference: {ref_timing:.3f}s")
        else:
            logger.warning(f"No reference timing available for bin_factor={bin_factor}")
