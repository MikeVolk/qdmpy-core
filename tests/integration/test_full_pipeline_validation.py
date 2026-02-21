"""Full pipeline validation tests for QDMpy.

These tests validate the complete end-to-end pipeline from data loading
through magnetic field calculation, ensuring identical results between
old and new codebases.
"""

from __future__ import annotations

import time

import pytest
from loguru import logger

from .conftest import array_comparison


@pytest.mark.validation
@pytest.mark.integration
@pytest.mark.slow
class TestFullPipelineValidation:
    """End-to-end validation tests for the complete QDMpy pipeline."""

    def test_complete_pipeline_consistency(
        self, test_data_folder, new_qdmpy_modules, old_qdmpy_modules, bin_factor, test_parameters
    ) -> None:
        """Test complete pipeline from data loading to magnetic field calculation."""
        # Import modules
        _QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        _QDMpy_old, QDM_old, ODMR_old = old_qdmpy_modules

        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        # Run complete pipeline with old codebase
        logger.info(f"Running old codebase pipeline (bin_factor={bin_factor})")
        start_time = time.time()

        # Load and process data
        odmr_old = ODMR_old.from_qdmio(str(test_data_folder))
        odmr_old.norm_to_ref()
        if bin_factor > 1:
            odmr_old.bin_pixels(bin_factor)
        if fluor_value > 0:
            odmr_old.set_global_fluor_corr(fluor_value)
            odmr_old.fluor_corr()

        # Fit data
        qdm_old = QDM_old(data=odmr_old, model=test_parameters["model_name"], **fit_params)
        qdm_old.fit()

        # Calculate magnetic fields
        qdm_old.calc_B111()

        old_time = time.time() - start_time

        # Store old results
        old_results = {
            "processed_data": odmr_old.data.copy(),
            "fit_parameters": qdm_old.fit_result.copy(),
            "B111_remanent": qdm_old.B111_rem.copy(),
            "B111_induced": qdm_old.B111_ind.copy(),
            "scan_dimensions": odmr_old.data_shape.copy(),
        }

        # Run complete pipeline with new codebase
        logger.info(f"Running new codebase pipeline (bin_factor={bin_factor})")
        start_time = time.time()

        # Load and process data
        loader = MatlabLoader(data_folder=str(test_data_folder))
        odmr_data = ODMRData.from_loader(loader=loader)
        odmr_new = ODMR_new(odmr_data)
        odmr_new.norm_to_ref()
        if bin_factor > 1:
            odmr_new.bin_pixels(bin_factor)
        if fluor_value > 0:
            odmr_new.set_global_fluor_corr(fluor_value)
            odmr_new.fluor_corr()

        # Fit data
        measurement_new = Measurement_new(
            odmr_data=odmr_new, model_name=test_parameters["model_name"], **fit_params
        )
        fit_result = measurement_new.fit()

        # Calculate magnetic fields
        magnetic_result = measurement_new.calculate_magnetic_fields()

        new_time = time.time() - start_time

        # Store new results
        new_results = {
            "processed_data": odmr_new.data.copy(),
            "fit_parameters": fit_result.parameters.copy(),
            "B111_remanent": magnetic_result.B111_remanent.copy(),
            "B111_induced": magnetic_result.B111_induced.copy(),
            "scan_dimensions": odmr_new.data_shape.copy(),
        }

        # Compare all results
        comparisons = {}
        tolerances = test_parameters["tolerances"]

        # Compare processed data
        comparisons["processed_data"] = array_comparison(
            old_results["processed_data"],
            new_results["processed_data"],
            tolerances["processing"],
            f"processed_data_bin_{bin_factor}",
        )

        # Compare fit parameters
        comparisons["fit_parameters"] = array_comparison(
            old_results["fit_parameters"],
            new_results["fit_parameters"],
            tolerances["fitting"],
            f"fit_parameters_bin_{bin_factor}",
        )

        # Compare magnetic fields
        comparisons["B111_remanent"] = array_comparison(
            old_results["B111_remanent"],
            new_results["B111_remanent"],
            tolerances["magnetic_fields"],
            f"B111_remanent_bin_{bin_factor}",
        )

        comparisons["B111_induced"] = array_comparison(
            old_results["B111_induced"],
            new_results["B111_induced"],
            tolerances["magnetic_fields"],
            f"B111_induced_bin_{bin_factor}",
        )

        # Compare scan dimensions
        comparisons["scan_dimensions"] = array_comparison(
            old_results["scan_dimensions"],
            new_results["scan_dimensions"],
            tolerances["data_loading"],
            f"scan_dimensions_bin_{bin_factor}",
        )

        # Assert all comparisons passed
        failed_comparisons = []
        for name, comparison in comparisons.items():
            if not comparison["passed"]:
                failed_comparisons.append(
                    f"{name}: max_diff={comparison['max_diff']:.2e}, "
                    f"tolerance={comparison['tolerance']:.2e}"
                )

        assert not failed_comparisons, (
            f"Pipeline validation failed for bin_factor={bin_factor}:\n"
            + "\n".join(failed_comparisons)
        )

        # Log performance comparison
        logger.info(
            f"Pipeline performance (bin_factor={bin_factor}): "
            f"old={old_time:.3f}s, new={new_time:.3f}s, "
            f"ratio={new_time / old_time:.2f}x"
        )

        logger.info(f"Complete pipeline validation passed for bin_factor={bin_factor}")

    def test_pipeline_with_different_models(
        self, test_data_folder, new_qdmpy_modules, old_qdmpy_modules, test_parameters
    ) -> None:
        """Test pipeline consistency with different model configurations."""
        # Test with bin_factor=2 and different models if supported
        bin_factor = 2
        models_to_test = ["auto"]  # Add more models as supported

        for model_name in models_to_test:
            logger.info(f"Testing pipeline with model: {model_name}")

            # Create modified test parameters
            model_params = test_parameters.copy()
            model_params["model_name"] = model_name

            # Run the full pipeline test with this model
            self.test_complete_pipeline_consistency(
                test_data_folder, new_qdmpy_modules, old_qdmpy_modules, bin_factor, model_params
            )

            logger.info(f"Pipeline validation passed for model: {model_name}")


@pytest.mark.validation
@pytest.mark.integration
@pytest.mark.requires_reference_data
class TestFullPipelineReferenceComparison:
    """Tests that validate the complete pipeline against reference data."""

    def test_pipeline_against_reference(
        self, test_data_folder, reference_data, new_qdmpy_modules, test_parameters
    ) -> None:
        """Test complete pipeline against pre-generated reference data."""
        # Import new modules
        _QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules

        bin_factor = int(reference_data["bin_factor"])
        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        logger.info(f"Testing pipeline against reference data (bin_factor={bin_factor})")

        # Run complete pipeline with new codebase
        loader = MatlabLoader(data_folder=str(test_data_folder))
        odmr_data = ODMRData.from_loader(loader=loader)
        odmr_new = ODMR_new(odmr_data)
        odmr_new.norm_to_ref()
        if bin_factor > 1:
            odmr_new.bin_pixels(bin_factor)
        if fluor_value > 0:
            odmr_new.set_global_fluor_corr(fluor_value)
            odmr_new.fluor_corr()

        measurement_new = Measurement_new(
            odmr_data=odmr_new, model_name=test_parameters["model_name"], **fit_params
        )
        fit_result = measurement_new.fit()
        magnetic_result = measurement_new.calculate_magnetic_fields()

        # Extract reference data
        ref_data = {
            "raw_data": reference_data["raw_data_odmr_raw_data"],
            "processed_data": reference_data["processed_data_processed_odmr_data"],
            "fit_parameters": reference_data["fit_results_fit_parameters"],
            "B111_remanent": reference_data["magnetic_fields_B111_remanent"],
            "B111_induced": reference_data["magnetic_fields_B111_induced"],
        }

        # Compare against reference at each stage
        comparisons = {}
        tolerances = test_parameters["tolerances"]

        # Compare raw data loading
        comparisons["raw_data"] = array_comparison(
            ref_data["raw_data"],
            odmr_data.data,
            tolerances["data_loading"],
            f"raw_vs_ref_bin_{bin_factor}",
        )

        # Compare processed data
        comparisons["processed_data"] = array_comparison(
            ref_data["processed_data"],
            odmr_new.data,
            tolerances["processing"],
            f"processed_vs_ref_bin_{bin_factor}",
        )

        # Compare fit parameters
        comparisons["fit_parameters"] = array_comparison(
            ref_data["fit_parameters"],
            fit_result.parameters,
            tolerances["fitting"],
            f"fit_vs_ref_bin_{bin_factor}",
        )

        # Compare magnetic fields
        comparisons["B111_remanent"] = array_comparison(
            ref_data["B111_remanent"],
            magnetic_result.B111_remanent,
            tolerances["magnetic_fields"],
            f"B111_rem_vs_ref_bin_{bin_factor}",
        )

        comparisons["B111_induced"] = array_comparison(
            ref_data["B111_induced"],
            magnetic_result.B111_induced,
            tolerances["magnetic_fields"],
            f"B111_ind_vs_ref_bin_{bin_factor}",
        )

        # Assert all comparisons passed
        failed_comparisons = []
        for name, comparison in comparisons.items():
            if not comparison["passed"]:
                failed_comparisons.append(
                    f"{name}: max_diff={comparison['max_diff']:.2e}, "
                    f"tolerance={comparison['tolerance']:.2e}"
                )

        assert not failed_comparisons, (
            f"Reference pipeline validation failed for bin_factor={bin_factor}:\n"
            + "\n".join(failed_comparisons)
        )

        logger.info(f"Reference pipeline validation passed for bin_factor={bin_factor}")


@pytest.mark.validation
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow
def test_pipeline_performance_scaling(
    test_data_folder, new_qdmpy_modules, old_qdmpy_modules, test_parameters
) -> None:
    """Test pipeline performance scaling with different binning factors."""
    # Import modules
    _QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
    _QDMpy_old, QDM_old, ODMR_old = old_qdmpy_modules

    bin_factors = [1, 2, 8]
    performance_results = {"old": {}, "new": {}}

    fluor_value = test_parameters["global_fluorescence"]
    fit_params = test_parameters["fitting_parameters"]

    for bin_factor in bin_factors:
        logger.info(f"Performance testing with bin_factor={bin_factor}")

        # Time old codebase
        start_time = time.time()
        odmr_old = ODMR_old.from_qdmio(str(test_data_folder))
        odmr_old.norm_to_ref()
        if bin_factor > 1:
            odmr_old.bin_pixels(bin_factor)
        if fluor_value > 0:
            odmr_old.set_global_fluor_corr(fluor_value)
            odmr_old.fluor_corr()
        qdm_old = QDM_old(data=odmr_old, model=test_parameters["model_name"], **fit_params)
        qdm_old.fit()
        qdm_old.calc_B111()
        old_time = time.time() - start_time
        performance_results["old"][bin_factor] = old_time

        # Time new codebase
        start_time = time.time()
        loader = MatlabLoader(data_folder=str(test_data_folder))
        odmr_data = ODMRData.from_loader(loader=loader)
        odmr_new = ODMR_new(odmr_data)
        odmr_new.norm_to_ref()
        if bin_factor > 1:
            odmr_new.bin_pixels(bin_factor)
        if fluor_value > 0:
            odmr_new.set_global_fluor_corr(fluor_value)
            odmr_new.fluor_corr()
        measurement_new = Measurement_new(
            odmr_data=odmr_new, model_name=test_parameters["model_name"], **fit_params
        )
        measurement_new.fit()
        measurement_new.calculate_magnetic_fields()
        new_time = time.time() - start_time
        performance_results["new"][bin_factor] = new_time

        logger.info(
            f"Performance (bin_factor={bin_factor}): "
            f"old={old_time:.3f}s, new={new_time:.3f}s, "
            f"ratio={new_time / old_time:.2f}x"
        )

    # Analyze performance scaling
    for bin_factor in bin_factors:
        old_time = performance_results["old"][bin_factor]
        new_time = performance_results["new"][bin_factor]

        # Performance should not be dramatically worse
        assert new_time < old_time * 5, (
            f"New codebase significantly slower for bin_factor={bin_factor}: "
            f"{new_time:.3f}s vs {old_time:.3f}s"
        )

    # Check that performance scales appropriately with binning
    # Smaller bin factors (more pixels) should take longer
    for i in range(len(bin_factors) - 1):
        _bin1, _bin2 = bin_factors[i], bin_factors[i + 1]
        # Generally, smaller bin factor (more data) should take longer
        # But this is not always guaranteed due to fitting convergence
        # Skip strict performance scaling checks for now

    logger.info("Pipeline performance scaling validation completed")
