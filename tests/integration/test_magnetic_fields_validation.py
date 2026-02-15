"""
Pytest validation tests for magnetic field calculation functionality.

These tests validate that the new QDMpy codebase calculates identical
magnetic field components (B111 remanent and induced) compared to the old codebase.
"""

import numpy as np
import pytest
from loguru import logger

from .conftest import array_comparison


@pytest.mark.validation
@pytest.mark.magnetic_fields
@pytest.mark.slow
class TestMagneticFieldValidation:
    """Validation tests for magnetic field calculations."""

    def test_b111_calculation_consistency(
        self, test_data_folder, new_qdmpy_modules, old_qdmpy_modules, bin_factor, test_parameters
    ):
        """Test that B111 magnetic field calculations are identical."""
        # Import modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
        QDMpy_old, QDM_old, ODMR_old = old_qdmpy_modules

        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        # Calculate with old codebase
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

        old_b111_rem = qdm_old.B111_rem.copy()
        old_b111_ind = qdm_old.B111_ind.copy()
        old_b111_total = (old_b111_rem + old_b111_ind).copy()

        # Calculate with new codebase
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

        new_b111_rem = magnetic_result.B111_remanent.copy()
        new_b111_ind = magnetic_result.B111_induced.copy()
        new_b111_total = (new_b111_rem + new_b111_ind).copy()

        # Compare B111 remanent
        b111_rem_comparison = array_comparison(
            old_b111_rem,
            new_b111_rem,
            test_parameters["tolerances"]["magnetic_fields"],
            f"B111_remanent_bin_{bin_factor}",
        )
        assert b111_rem_comparison["passed"], (
            f"B111 remanent mismatch (bin={bin_factor}): "
            f"max_diff={b111_rem_comparison['max_diff']:.2e}, "
            f"tolerance={b111_rem_comparison['tolerance']:.2e}"
        )

        # Compare B111 induced
        b111_ind_comparison = array_comparison(
            old_b111_ind,
            new_b111_ind,
            test_parameters["tolerances"]["magnetic_fields"],
            f"B111_induced_bin_{bin_factor}",
        )
        assert b111_ind_comparison["passed"], (
            f"B111 induced mismatch (bin={bin_factor}): "
            f"max_diff={b111_ind_comparison['max_diff']:.2e}, "
            f"tolerance={b111_ind_comparison['tolerance']:.2e}"
        )

        # Compare B111 total
        b111_total_comparison = array_comparison(
            old_b111_total,
            new_b111_total,
            test_parameters["tolerances"]["magnetic_fields"],
            f"B111_total_bin_{bin_factor}",
        )
        assert b111_total_comparison["passed"], (
            f"B111 total mismatch (bin={bin_factor}): "
            f"max_diff={b111_total_comparison['max_diff']:.2e}, "
            f"tolerance={b111_total_comparison['tolerance']:.2e}"
        )

        logger.info(f"B111 magnetic field validation passed for bin_factor={bin_factor}")

    def test_magnetic_field_properties(
        self, test_data_folder, new_qdmpy_modules, bin_factor, test_parameters
    ):
        """Test that magnetic field results have expected properties."""
        # Import new modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules

        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        # Calculate with new codebase
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

        # Check field shapes are consistent
        assert (
            magnetic_result.B111_remanent.shape == magnetic_result.B111_induced.shape
        ), "B111 remanent and induced should have same shape"

        # Check fields are finite where valid
        valid_mask = ~np.isnan(magnetic_result.B111_remanent)
        if np.any(valid_mask):
            assert np.all(
                np.isfinite(magnetic_result.B111_remanent[valid_mask])
            ), "B111 remanent should be finite where not NaN"
            assert np.all(
                np.isfinite(magnetic_result.B111_induced[valid_mask])
            ), "B111 induced should be finite where not NaN"

        # Check field magnitudes are reasonable (within Tesla range)
        b111_total = magnetic_result.B111_remanent + magnetic_result.B111_induced
        if np.any(valid_mask):
            max_field = np.max(np.abs(b111_total[valid_mask]))
            assert max_field < 10.0, f"Magnetic field magnitude unreasonable: {max_field:.2e} T"

        logger.info(f"Magnetic field properties validation passed for bin_factor={bin_factor}")


@pytest.mark.validation
@pytest.mark.magnetic_fields
@pytest.mark.requires_reference_data
class TestMagneticFieldReferenceComparison:
    """Tests that compare magnetic field calculations against pre-generated reference data."""

    def test_against_reference_magnetic_fields(
        self, test_data_folder, reference_data, new_qdmpy_modules, test_parameters
    ):
        """Test magnetic field calculations against reference data."""
        # Import new modules
        QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules

        bin_factor = int(reference_data["bin_factor"])
        fluor_value = test_parameters["global_fluorescence"]
        fit_params = test_parameters["fitting_parameters"]

        # Calculate with new codebase
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

        # Get reference magnetic fields
        ref_b111_rem = reference_data["magnetic_fields_B111_remanent"]
        ref_b111_ind = reference_data["magnetic_fields_B111_induced"]
        ref_b111_total = reference_data["magnetic_fields_B111_total"]

        # Compare B111 remanent against reference
        b111_rem_comparison = array_comparison(
            ref_b111_rem,
            magnetic_result.B111_remanent,
            test_parameters["tolerances"]["magnetic_fields"],
            f"B111_rem_vs_reference_bin_{bin_factor}",
        )
        assert b111_rem_comparison["passed"], (
            f"B111 remanent differs from reference (bin={bin_factor}): "
            f"max_diff={b111_rem_comparison['max_diff']:.2e}, "
            f"tolerance={b111_rem_comparison['tolerance']:.2e}"
        )

        # Compare B111 induced against reference
        b111_ind_comparison = array_comparison(
            ref_b111_ind,
            magnetic_result.B111_induced,
            test_parameters["tolerances"]["magnetic_fields"],
            f"B111_ind_vs_reference_bin_{bin_factor}",
        )
        assert b111_ind_comparison["passed"], (
            f"B111 induced differs from reference (bin={bin_factor}): "
            f"max_diff={b111_ind_comparison['max_diff']:.2e}, "
            f"tolerance={b111_ind_comparison['tolerance']:.2e}"
        )

        # Compare B111 total against reference
        new_b111_total = magnetic_result.B111_remanent + magnetic_result.B111_induced
        b111_total_comparison = array_comparison(
            ref_b111_total,
            new_b111_total,
            test_parameters["tolerances"]["magnetic_fields"],
            f"B111_total_vs_reference_bin_{bin_factor}",
        )
        assert b111_total_comparison["passed"], (
            f"B111 total differs from reference (bin={bin_factor}): "
            f"max_diff={b111_total_comparison['max_diff']:.2e}, "
            f"tolerance={b111_total_comparison['tolerance']:.2e}"
        )

        logger.info(f"Reference magnetic field comparison passed for bin_factor={bin_factor}")


@pytest.mark.validation
@pytest.mark.magnetic_fields
@pytest.mark.slow
@pytest.mark.performance
def test_magnetic_field_performance(
    test_data_folder, new_qdmpy_modules, old_qdmpy_modules, bin_factor, test_parameters
):
    """Performance test for magnetic field calculations."""
    import time

    # Import modules
    QDMpy_new, Measurement_new, ODMR_new, ODMRData, MatlabLoader = new_qdmpy_modules
    QDMpy_old, QDM_old, ODMR_old = old_qdmpy_modules

    fluor_value = test_parameters["global_fluorescence"]
    fit_params = test_parameters["fitting_parameters"]

    # Time old codebase magnetic field calculation
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

    # Time new codebase magnetic field calculation
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
    magnetic_result = measurement_new.calculate_magnetic_fields()
    new_time = time.time() - start_time

    logger.info(
        f"Magnetic field performance (bin={bin_factor}): old={old_time:.3f}s, new={new_time:.3f}s"
    )

    # Performance should be reasonable
    assert new_time < old_time * 3, (
        f"New codebase is significantly slower (bin={bin_factor}): "
        f"{new_time:.3f}s vs {old_time:.3f}s"
    )
