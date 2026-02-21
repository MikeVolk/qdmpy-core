#!/usr/bin/env python3
"""Reference data generation script for QDMpy validation.

This script generates reference results using the old QDMpy codebase for multiple
binning factors. The reference data is used by the pytest validation suite to
ensure the new codebase produces identical results.

Usage:
    python scripts/generate_reference_data.py [--data-folder PATH] [--output-dir PATH]

The script processes data with binning factors 1, 2, and 8, generating:
- Raw ODMR data after loading
- Processed data after normalization and binning
- Fit results and parameters
- Magnetic field calculations (B111 components)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# Import utilities for old codebase access
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from validation_tests.utils import safe_import_old_qdmpy
except ImportError:
    logger.error("Could not import validation utilities. Ensure validation_tests/ is present.")
    sys.exit(1)


class ReferenceDataGenerator:
    """Generates reference data using the old QDMpy codebase."""

    def __init__(self, data_folder: str, output_dir: str) -> None:
        """Initialize the reference data generator.

        Args:
            data_folder: Path to test data containing .mat files and reference images
            output_dir: Directory to save reference data files
        """
        self.data_folder = Path(data_folder)
        self.output_dir = Path(output_dir)
        self.binning_factors = [1, 2, 8]
        self.test_parameters = {
            "global_fluorescence": 0.2,
            "model_name": "auto",
            "pixel_spacing": 4e-6,
            "fitting_parameters": {"max_iterations": 1000, "tolerance": 1e-6, "estimator": "LSE"},
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Reference data generator initialized:")
        logger.info(f"  Data folder: {self.data_folder}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Binning factors: {self.binning_factors}")

    def validate_data_folder(self) -> bool:
        """Validate that the data folder contains required files."""
        required_files = ["run_00000.mat", "run_00001.mat", "LED.csv", "laser.csv"]
        missing_files = []

        for filename in required_files:
            if not (self.data_folder / filename).exists():
                missing_files.append(filename)

        if missing_files:
            logger.error(f"Missing required files in {self.data_folder}: {missing_files}")
            return False

        logger.info("Data folder validation passed - all required files present")
        return True

    def generate_reference_data(self) -> bool:
        """Generate reference data for all binning factors.

        Returns:
            True if all reference data generated successfully, False otherwise
        """
        logger.info("Starting reference data generation")
        start_time = time.time()

        # Validate data folder
        if not self.validate_data_folder():
            return False

        # Import old codebase
        QDMpy_old, QDM_old, ODMR_old = safe_import_old_qdmpy()
        if QDM_old is None:
            logger.error("Failed to import old QDMpy codebase")
            return False

        success = True

        for bin_factor in self.binning_factors:
            logger.info(f"Generating reference data for binning factor {bin_factor}")

            try:
                # Generate reference data for this binning factor
                reference_data = self._generate_for_binning(
                    QDMpy_old, QDM_old, ODMR_old, bin_factor
                )

                if reference_data is None:
                    logger.error(
                        f"Failed to generate reference data for binning factor {bin_factor}"
                    )
                    success = False
                    continue

                # Save reference data
                output_file = self.output_dir / f"reference_bin_{bin_factor}.npz"
                self._save_reference_data(reference_data, output_file)
                logger.info(f"Saved reference data: {output_file}")

            except Exception as e:
                logger.exception(
                    f"Error generating reference data for binning factor {bin_factor}: {e}"
                )
                success = False

        total_time = time.time() - start_time
        logger.info(f"Reference data generation completed in {total_time:.2f} seconds")
        logger.info(f"Success: {success}")

        return success

    def _generate_for_binning(
        self, QDMpy_old: Any, QDM_old: Any, ODMR_old: Any, bin_factor: int
    ) -> dict[str, Any] | None:
        """Generate reference data for a specific binning factor.

        Args:
            QDMpy_old: Old QDMpy module
            QDM_old: Old QDM class
            ODMR_old: Old ODMR class
            bin_factor: Binning factor to apply

        Returns:
            Dictionary containing all reference data or None if failed
        """
        try:
            # Stage 1: Load raw data
            logger.debug(f"Loading raw data (bin_factor={bin_factor})")
            raw_data = self._load_raw_data(ODMR_old)
            if raw_data is None:
                return None

            # Stage 2: Process data (normalize, bin, fluorescence correction)
            logger.debug(f"Processing data (bin_factor={bin_factor})")
            processed_data = self._process_data(raw_data, bin_factor)
            if processed_data is None:
                return None

            # Stage 3: Fit data
            logger.debug(f"Fitting data (bin_factor={bin_factor})")
            fit_results = self._fit_data(QDM_old, processed_data)
            if fit_results is None:
                return None

            # Stage 4: Calculate magnetic fields
            logger.debug(f"Calculating magnetic fields (bin_factor={bin_factor})")
            magnetic_fields = self._calculate_magnetic_fields(fit_results)
            if magnetic_fields is None:
                return None

            # Package all reference data
            reference_data = {
                "bin_factor": bin_factor,
                "timestamp": time.time(),
                "test_parameters": self.test_parameters.copy(),
                "raw_data": raw_data,
                "processed_data": processed_data,
                "fit_results": fit_results,
                "magnetic_fields": magnetic_fields,
            }

            logger.debug(f"Reference data generated successfully for bin_factor={bin_factor}")
            return reference_data

        except Exception as e:
            logger.exception(f"Failed to generate reference data for bin_factor={bin_factor}: {e}")
            return None

    def _load_raw_data(self, ODMR_old: Any) -> dict[str, Any] | None:
        """Load raw ODMR data using old codebase.

        Args:
            ODMR_old: Old ODMR class

        Returns:
            Dictionary containing raw data or None if failed
        """
        try:
            # Load ODMR data
            odmr = ODMR_old.from_qdmio(str(self.data_folder))

            # Load reference images
            led_image = np.genfromtxt(self.data_folder / "LED.csv", delimiter=",")
            laser_image = np.genfromtxt(self.data_folder / "laser.csv", delimiter=",")

            raw_data = {
                "odmr_raw_data": odmr._raw_data.copy(),
                "frequencies": odmr.frequencies.copy(),
                "scan_dimensions": odmr.data_shape.copy(),
                "led_image": led_image.copy(),
                "laser_image": laser_image.copy(),
                "metadata": {
                    "n_pol": odmr.n_pol,
                    "n_frange": odmr.n_frange,
                    "n_freqs": odmr.n_freqs,
                },
            }

            logger.debug(f"Raw data loaded: shape={raw_data['odmr_raw_data'].shape}")
            return raw_data

        except Exception as e:
            logger.exception(f"Failed to load raw data: {e}")
            return None

    def _process_data(self, raw_data: dict[str, Any], bin_factor: int) -> dict[str, Any] | None:
        """Process raw data with the old codebase.

        Args:
            raw_data: Raw ODMR data
            bin_factor: Binning factor to apply

        Returns:
            Dictionary containing processed data or None if failed
        """
        try:
            # Import old processing modules
            _QDMpy_old, _QDM_old, ODMR_old = safe_import_old_qdmpy()

            # Recreate ODMR object from raw data
            odmr = ODMR_old.from_qdmio(str(self.data_folder))

            # Apply normalization
            odmr.norm_to_ref()

            # Apply binning if requested
            if bin_factor > 1:
                odmr.bin_pixels(bin_factor)

            # Apply fluorescence correction
            if self.test_parameters["global_fluorescence"] > 0:
                odmr.set_global_fluor_corr(self.test_parameters["global_fluorescence"])
                odmr.fluor_corr()

            processed_data = {
                "processed_odmr_data": odmr.data.copy(),
                "frequencies": odmr.frequencies.copy(),
                "scan_dimensions": odmr.data_shape.copy(),
                "bin_factor": bin_factor,
                "normalization_applied": True,
                "fluorescence_correction": self.test_parameters["global_fluorescence"],
                "metadata": {
                    "data_shape": odmr.data.shape,
                    "scan_shape": odmr.data_shape,
                    "n_pol": odmr.n_pol,
                    "n_frange": odmr.n_frange,
                    "n_freqs": odmr.n_freqs,
                },
            }

            logger.debug(f"Data processed: shape={processed_data['processed_odmr_data'].shape}")
            return processed_data

        except Exception as e:
            logger.exception(f"Failed to process data: {e}")
            return None

    def _fit_data(self, QDM_old: Any, processed_data: dict[str, Any]) -> dict[str, Any] | None:
        """Fit processed data using old codebase.

        Args:
            QDM_old: Old QDM class
            processed_data: Processed ODMR data

        Returns:
            Dictionary containing fit results or None if failed
        """
        try:
            # Import old fitting modules
            _QDMpy_old, QDM_old, ODMR_old = safe_import_old_qdmpy()

            # Recreate processed ODMR object
            odmr = ODMR_old.from_qdmio(str(self.data_folder))
            odmr.norm_to_ref()

            if processed_data["bin_factor"] > 1:
                odmr.bin_pixels(processed_data["bin_factor"])

            if processed_data["fluorescence_correction"] > 0:
                odmr.set_global_fluor_corr(processed_data["fluorescence_correction"])
                odmr.fluor_corr()

            # Create QDM measurement object
            qdm_measurement = QDM_old(
                data=odmr,
                model=self.test_parameters["model_name"],
                **self.test_parameters["fitting_parameters"],
            )

            # Perform fitting
            qdm_measurement.fit()

            fit_results = {
                "fit_parameters": qdm_measurement.fit_result.copy(),
                "fit_errors": qdm_measurement.fit_errors.copy()
                if hasattr(qdm_measurement, "fit_errors")
                else None,
                "fit_quality": {
                    "chi_squared": qdm_measurement.chi_sq.copy()
                    if hasattr(qdm_measurement, "chi_sq")
                    else None,
                    "iterations": qdm_measurement.iterations.copy()
                    if hasattr(qdm_measurement, "iterations")
                    else None,
                },
                "model_info": {
                    "model_name": self.test_parameters["model_name"],
                    "n_params": qdm_measurement.fit_result.shape[-1]
                    if qdm_measurement.fit_result is not None
                    else 0,
                },
                "metadata": {
                    "fit_shape": qdm_measurement.fit_result.shape
                    if qdm_measurement.fit_result is not None
                    else None,
                    "fitting_parameters": self.test_parameters["fitting_parameters"],
                },
            }

            logger.debug(f"Fitting completed: fit_shape={fit_results['metadata']['fit_shape']}")
            return fit_results

        except Exception as e:
            logger.exception(f"Failed to fit data: {e}")
            return None

    def _calculate_magnetic_fields(self, fit_results: dict[str, Any]) -> dict[str, Any] | None:
        """Calculate magnetic fields from fit results using old codebase.

        Args:
            fit_results: Fit results from old codebase

        Returns:
            Dictionary containing magnetic field calculations or None if failed
        """
        try:
            # Import old codebase
            _QDMpy_old, QDM_old, ODMR_old = safe_import_old_qdmpy()

            # Recreate the full measurement pipeline
            odmr = ODMR_old.from_qdmio(str(self.data_folder))
            odmr.norm_to_ref()

            if fit_results["metadata"]["fitting_parameters"].get("bin_factor", 1) > 1:
                bin_factor = fit_results["metadata"]["fitting_parameters"]["bin_factor"]
                odmr.bin_pixels(bin_factor)

            if self.test_parameters["global_fluorescence"] > 0:
                odmr.set_global_fluor_corr(self.test_parameters["global_fluorescence"])
                odmr.fluor_corr()

            qdm_measurement = QDM_old(
                data=odmr,
                model=self.test_parameters["model_name"],
                **self.test_parameters["fitting_parameters"],
            )
            qdm_measurement.fit()

            # Calculate magnetic fields
            qdm_measurement.calc_B111()

            magnetic_fields = {
                "B111_remanent": qdm_measurement.B111_rem.copy()
                if hasattr(qdm_measurement, "B111_rem")
                else None,
                "B111_induced": qdm_measurement.B111_ind.copy()
                if hasattr(qdm_measurement, "B111_ind")
                else None,
                "B111_total": (qdm_measurement.B111_rem + qdm_measurement.B111_ind).copy()
                if hasattr(qdm_measurement, "B111_rem") and hasattr(qdm_measurement, "B111_ind")
                else None,
                "calculation_parameters": {"pixel_spacing": self.test_parameters["pixel_spacing"]},
                "metadata": {
                    "field_shape": qdm_measurement.B111_rem.shape
                    if hasattr(qdm_measurement, "B111_rem")
                    else None
                },
            }

            logger.debug(
                f"Magnetic fields calculated: shape={magnetic_fields['metadata']['field_shape']}"
            )
            return magnetic_fields

        except Exception as e:
            logger.exception(f"Failed to calculate magnetic fields: {e}")
            return None

    def _save_reference_data(self, reference_data: dict[str, Any], output_file: Path) -> None:
        """Save reference data to compressed numpy file.

        Args:
            reference_data: Complete reference data dictionary
            output_file: Path to save reference data
        """
        try:
            # Prepare data for saving (numpy arrays only)
            save_dict = {}

            # Flatten nested dictionaries with prefixed keys
            for stage_key, stage_data in reference_data.items():
                if isinstance(stage_data, dict):
                    for data_key, data_value in stage_data.items():
                        if isinstance(data_value, dict):
                            for sub_key, sub_value in data_value.items():
                                if isinstance(sub_value, np.ndarray):
                                    save_dict[f"{stage_key}_{data_key}_{sub_key}"] = sub_value
                                elif isinstance(sub_value, (int, float, str, bool)):
                                    save_dict[f"{stage_key}_{data_key}_{sub_key}"] = np.array(
                                        sub_value
                                    )
                        elif isinstance(data_value, np.ndarray):
                            save_dict[f"{stage_key}_{data_key}"] = data_value
                        elif isinstance(data_value, (int, float, str, bool)):
                            save_dict[f"{stage_key}_{data_key}"] = np.array(data_value)
                elif isinstance(stage_data, np.ndarray):
                    save_dict[stage_key] = stage_data
                elif isinstance(stage_data, (int, float, str, bool)):
                    save_dict[stage_key] = np.array(stage_data)

            # Save to compressed file
            np.savez_compressed(output_file, **save_dict)

            logger.debug(f"Reference data saved to {output_file}")
            logger.debug(f"Saved {len(save_dict)} data arrays")

        except Exception as e:
            logger.exception(f"Failed to save reference data: {e}")
            raise


def main() -> int | None:
    """Main entry point for reference data generation."""
    parser = argparse.ArgumentParser(
        description="Generate reference data using old QDMpy codebase for validation"
    )
    parser.add_argument(
        "--data-folder",
        default="/home/mike/Documents/FOV18x",
        help="Path to test data folder (default: /home/mike/Documents/FOV18x)",
    )
    parser.add_argument(
        "--output-dir",
        default="./reference_data",
        help="Output directory for reference data (default: ./reference_data)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not Path(args.data_folder).exists():
        logger.error(f"Data folder does not exist: {args.data_folder}")
        return 1

    try:
        # Initialize generator
        generator = ReferenceDataGenerator(args.data_folder, args.output_dir)

        # Generate reference data
        success = generator.generate_reference_data()

        if success:
            return 0
        return 1

    except Exception as e:
        logger.exception(f"Reference data generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
