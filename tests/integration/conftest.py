"""
Pytest configuration and fixtures for integration tests.

This module provides shared fixtures and configuration for integration tests
that validate the new QDMpy codebase against reference data generated from
the old codebase.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

# Add project root to path for importing validation utilities
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation utilities if available
try:
    from validation_tests.utils import safe_import_new_qdmpy

    VALIDATION_UTILS_AVAILABLE = True
except ImportError:
    VALIDATION_UTILS_AVAILABLE = False


@pytest.fixture(scope="session")
def test_data_folder():
    """Fixture providing path to test data folder."""
    # Use FOV18x since we have reference data for it
    data_folder = PROJECT_ROOT / "tests" / "data" / "FOV18x"
    if not data_folder.exists():
        pytest.skip(f"Test data folder not found: {data_folder}")
    return data_folder


@pytest.fixture(scope="session")
def reference_data_folder():
    """Fixture providing path to reference data folder."""
    ref_folder = PROJECT_ROOT / "reference_data"
    if not ref_folder.exists():
        pytest.skip(f"Reference data folder not found: {ref_folder}")
    return ref_folder


@pytest.fixture(scope="session")
def new_qdmpy_modules():
    """Fixture providing new QDMpy modules."""
    if not VALIDATION_UTILS_AVAILABLE:
        pytest.skip("Validation utilities not available")

    imports = safe_import_new_qdmpy()
    if imports[0] is None:
        pytest.skip("Failed to import new QDMpy modules")

    return imports


# Note: old_qdmpy_modules fixture removed since we use reference data instead


@pytest.fixture(params=[1, 2, 8])
def bin_factor(request):
    """Fixture providing different binning factors for testing."""
    return request.param


@pytest.fixture
def reference_data(reference_data_folder, test_data_folder, bin_factor):
    """Fixture providing reference data for a specific dataset and binning factor."""
    # Determine reference filename based on test data folder
    dataset_name = test_data_folder.name
    ref_file = reference_data_folder / f"{dataset_name}_reference_bin{bin_factor}.npz"

    if not ref_file.exists():
        pytest.skip(f"Reference data not found: {ref_file}")

    try:
        with np.load(ref_file, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    except Exception as e:
        pytest.skip(f"Failed to load reference data: {e}")


@pytest.fixture
def test_parameters():
    """Fixture providing standard test parameters."""
    return {
        "global_fluorescence": 0.2,
        "model_name": "auto",
        "pixel_spacing": 4e-6,
        "fitting_parameters": {"max_iterations": 1000, "tolerance": 1e-6, "estimator": "LSE"},
        "tolerances": {
            "data_loading": 1e-15,
            "processing": 1e-12,
            "fitting": 1e-8,
            "magnetic_fields": 1e-6,
        },
    }


def array_comparison(
    array1: np.ndarray, array2: np.ndarray, tolerance: float, name: str = "array"
) -> Dict[str, Any]:
    """Compare two arrays with detailed statistics.

    Args:
        array1: First array
        array2: Second array
        tolerance: Tolerance for comparison
        name: Name for logging

    Returns:
        Dictionary with comparison results
    """
    # Check shapes
    if array1.shape != array2.shape:
        return {
            "passed": False,
            "error": f"Shape mismatch: {array1.shape} vs {array2.shape}",
            "name": name,
        }

    # Handle NaN values
    valid_mask = ~(np.isnan(array1) | np.isnan(array2))
    if not np.any(valid_mask):
        return {
            "passed": True,
            "max_diff": 0.0,
            "mean_diff": 0.0,
            "name": name,
            "note": "All values are NaN",
        }

    # Calculate differences only for valid values
    diff = np.abs(array1[valid_mask] - array2[valid_mask])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    passed = max_diff <= tolerance

    return {
        "passed": passed,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tolerance,
        "name": name,
        "valid_pixels": np.sum(valid_mask),
        "total_pixels": array1.size,
    }
