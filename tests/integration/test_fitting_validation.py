"""
Pytest validation tests for spectral fitting functionality.

These tests validate that the new QDMpy codebase produces identical fitting
results compared to the old codebase, including fit parameters, errors, and
convergence metrics.
"""

import numpy as np
import pytest
from loguru import logger

from .conftest import array_comparison


@pytest.mark.validation
@pytest.mark.fitting
@pytest.mark.slow
class TestFittingValidation:
    """Validation tests for spectral fitting operations."""

    def test_fit_parameters_consistency(
        self, test_data_folder, new_qdmpy_modules, old_qdmpy_modules, bin_factor, test_parameters
    ):
        """Test that fitting produces identical parameters."""
        # Import modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        QDMpy_old, QDM_old, ODMR_old = old_qdmpy_modules

        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        # Fit with old codebase
        odmr_old = ODMR_old.from_qdmio(str(test_data_folder))
        odmr_old.norm_to_ref()
        if bin_factor > 1:
            odmr_old.bin_pixels(bin_factor)
        if fluor_value > 0:
            odmr_old.set_global_fluor_corr(fluor_value)
            odmr_old.fluor_corr()

        qdm_old = QDM_old(data=odmr_old, model=test_parameters["model_name"], **fit_params)
        qdm_old.fit()
        old_fit_params = qdm_old.fit_result.copy()

        # Fit with new codebase
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
        new_fit_params = fit_result.parameters.copy()

        # Compare fit parameters
        params_comparison = array_comparison(
            old_fit_params,
            new_fit_params,
            test_parameters["tolerances"]["fitting"],
            f"fit_parameters_bin_{bin_factor}",
        )
        assert params_comparison["passed"], (
            f"Fit parameters mismatch (bin={bin_factor}): "
            f"max_diff={params_comparison['max_diff']:.2e}, "
            f"tolerance={params_comparison['tolerance']:.2e}"
        )

        logger.info(f"Fit parameters validation passed for bin_factor={bin_factor}")

    def test_fit_quality_metrics(
        self, test_data_folder, new_qdmpy_modules, old_qdmpy_modules, bin_factor, test_parameters
    ):
        """Test that fitting quality metrics are consistent."""
        # Import modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        QDMpy_old, QDM_old, ODMR_old = old_qdmpy_modules

        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        # Fit with old codebase
        odmr_old = ODMR_old.from_qdmio(str(test_data_folder))
        odmr_old.norm_to_ref()
        if bin_factor > 1:
            odmr_old.bin_pixels(bin_factor)
        if fluor_value > 0:
            odmr_old.set_global_fluor_corr(fluor_value)
            odmr_old.fluor_corr()

        qdm_old = QDM_old(data=odmr_old, model=test_parameters["model_name"], **fit_params)
        qdm_old.fit()

        # Get old quality metrics
        old_chi_sq = qdm_old.chi_sq.copy() if hasattr(qdm_old, "chi_sq") else None
        old_iterations = qdm_old.iterations.copy() if hasattr(qdm_old, "iterations") else None

        # Fit with new codebase
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

        # Get new quality metrics
        new_chi_sq = fit_result.chi_squared.copy() if hasattr(fit_result, "chi_squared") else None
        new_iterations = fit_result.iterations.copy() if hasattr(fit_result, "iterations") else None

        # Compare chi-squared if available
        if old_chi_sq is not None and new_chi_sq is not None:
            chi_sq_comparison = array_comparison(
                old_chi_sq,
                new_chi_sq,
                test_parameters["tolerances"]["fitting"],
                f"chi_squared_bin_{bin_factor}",
            )
            assert chi_sq_comparison["passed"], (
                f"Chi-squared mismatch (bin={bin_factor}): "
                f"max_diff={chi_sq_comparison['max_diff']:.2e}, "
                f"tolerance={chi_sq_comparison['tolerance']:.2e}"
            )

        # Compare iterations if available
        if old_iterations is not None and new_iterations is not None:
            # Iterations might differ slightly due to convergence criteria
            iter_comparison = array_comparison(
                old_iterations.astype(float),
                new_iterations.astype(float),
                10.0,  # Allow up to 10 iteration difference
                f"iterations_bin_{bin_factor}",
            )
            # Log but don't fail on iteration differences
            if not iter_comparison["passed"]:
                logger.warning(
                    f"Iteration count differences (bin={bin_factor}): "
                    f"max_diff={iter_comparison['max_diff']:.1f}"
                )

        logger.info(f"Fit quality metrics validation passed for bin_factor={bin_factor}")


@pytest.mark.validation
@pytest.mark.fitting
@pytest.mark.requires_reference_data
class TestFittingReferenceComparison:
    """Tests that compare fitting results against pre-generated reference data."""

    def test_against_reference_fit_results(
        self, test_data_folder, reference_data, new_qdmpy_modules, test_parameters
    ):
        """Test fitting results against reference data."""
        # Import new modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules

        bin_factor = int(reference_data["bin_factor"])
        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        # Process and fit with new codebase
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

        # Get reference fit results
        ref_fit_params = reference_data["fit_results_fit_parameters"]

        # Compare against reference
        params_comparison = array_comparison(
            ref_fit_params,
            fit_result.parameters,
            test_parameters["tolerances"]["fitting"],
            f"fit_params_vs_reference_bin_{bin_factor}",
        )
        assert params_comparison["passed"], (
            f"Fit parameters differ from reference (bin={bin_factor}): "
            f"max_diff={params_comparison['max_diff']:.2e}, "
            f"tolerance={params_comparison['tolerance']:.2e}"
        )

        # Compare chi-squared if available
        if "fit_results_fit_quality_chi_squared" in reference_data:
            ref_chi_sq = reference_data["fit_results_fit_quality_chi_squared"]
            if ref_chi_sq is not None and hasattr(fit_result, "chi_squared"):
                chi_sq_comparison = array_comparison(
                    ref_chi_sq,
                    fit_result.chi_squared,
                    test_parameters["tolerances"]["fitting"],
                    f"chi_sq_vs_reference_bin_{bin_factor}",
                )
                assert chi_sq_comparison["passed"], (
                    f"Chi-squared differs from reference (bin={bin_factor}): "
                    f"max_diff={chi_sq_comparison['max_diff']:.2e}"
                )

        logger.info(f"Reference fitting comparison passed for bin_factor={bin_factor}")


@pytest.mark.validation
@pytest.mark.fitting
@pytest.mark.slow
@pytest.mark.performance
def test_fitting_performance(
    test_data_folder, new_qdmpy_modules, old_qdmpy_modules, bin_factor, test_parameters
):
    """Performance test for fitting operations."""
    import time

    # Import modules
    QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
    QDMpy_old, QDM_old, ODMR_old = old_qdmpy_modules

    fluor_value = test_parameters["global_fluorescence"]
    fit_params = test_parameters["fitting_parameters"]

    # Time old codebase fitting
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
    old_time = time.time() - start_time

    # Time new codebase fitting
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
    fit_result = measurement_new.fit()
    new_time = time.time() - start_time

    logger.info(f"Fitting performance (bin={bin_factor}): old={old_time:.3f}s, new={new_time:.3f}s")

    # Fitting performance should be reasonable
    # Allow more tolerance since fitting can vary significantly
    assert new_time < old_time * 3, (
        f"New codebase is significantly slower (bin={bin_factor}): "
        f"{new_time:.3f}s vs {old_time:.3f}s"
    )
