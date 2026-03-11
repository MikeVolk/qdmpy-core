# Integration Validation Tests

This directory contains integration tests that validate the new QDMpy codebase produces identical results to reference data generated from the old codebase.

## Overview

The integration tests validate:
- Data loading equivalence against reference data
- Processing pipeline consistency (normalization, binning)
- End-to-end pipeline validation across multiple binning factors

## Two-Phase Validation Approach

### Phase 1: Generate Reference Data (Run Once)
Generate reference data using the old codebase:
```bash
uv run python simple_reference_generator.py --data-folder tests/data/FOV18x --output-dir reference_data
```

This creates:
- `FOV18x_reference_bin1.npz` - Reference data with no binning
- `FOV18x_reference_bin2.npz` - Reference data with 2x binning
- `FOV18x_reference_bin8.npz` - Reference data with 8x binning

### Phase 2: Run Validation Tests (Use Forever)
Tests load reference `.npz` files and compare new codebase results:
```bash
uv run pytest tests/integration/ -m validation --no-cov
```

**No old codebase needed after reference data generation!**

## Test Structure

### Fixtures (`conftest.py`)
- `test_data_folder`: Provides path to test data (FOV18x)
- `reference_data_folder`: Provides path to reference data
- `new_qdmpy_modules`: Imports new QDMpy modules
- `bin_factor`: Parametrizes tests across binning factors [1, 2, 8]
- `reference_data`: Loads reference data for specific dataset and binning factor
- `test_parameters`: Standard test parameters and tolerances

### Test Categories

#### Data Loading Tests (`test_data_loading_validation.py`)
- Raw ODMR data loading validation against reference
- Reference image loading validation
- Frequency and scan dimension validation

#### Processing Tests (`test_processing_validation.py`)
- Normalization consistency with reference
- Binning accuracy validation
- Performance benchmarking (if reference timing available)

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.validation`: All validation tests
- `@pytest.mark.slow`: Tests that take >10 seconds
- `@pytest.mark.performance`: Performance benchmarking tests
- `@pytest.mark.data_loading`: Data loading specific tests
- `@pytest.mark.processing`: Processing pipeline tests
- `@pytest.mark.binning`: Binning specific tests

## Usage

### Run All Validation Tests
```bash
uv run pytest tests/integration/ -m validation --no-cov
```

### Run Specific Test Categories
```bash
# Data loading only
uv run pytest tests/integration/ -m data_loading --no-cov

# Processing pipeline only
uv run pytest tests/integration/ -m processing --no-cov

# Fast tests only (exclude slow tests)
uv run pytest tests/integration/ -m "validation and not slow" --no-cov
```

### Run Tests for Specific Binning Factor
```bash
# Test only binning factor 2
uv run pytest tests/integration/ -k "bin_2" --no-cov

# Test binning-specific functionality
uv run pytest tests/integration/ -m binning --no-cov
```

## Requirements

- New QDMpy codebase properly installed
- Reference data files (`.npz`) in `reference_data/` folder
- Test data in `tests/data/FOV18x/` folder

## Test Data

Tests expect the following data structure:
```
tests/data/FOV18x/
├── run_00000.mat
├── run_00001.mat
├── LED.csv
└── laser.csv

reference_data/
├── FOV18x_reference_bin1.npz
├── FOV18x_reference_bin2.npz
└── FOV18x_reference_bin8.npz
```

## Reference Data Contents

Each `.npz` file contains:
- `raw_data`: Raw ODMR data after loading
- `frequencies`: Frequency arrays
- `scan_dimensions`: Scan dimensions
- `normalized_data`: Data after normalization
- `binned_data`: Data after binning (if applicable)
- `binned_scan_dimensions`: Scan dimensions after binning
- Various metadata and shape information

## Validation Tolerances

Different validation stages use appropriate tolerances:
- Data loading: exact match for raw data, 1e-6 relative tolerance for frequencies
- Processing: 1e-12 relative tolerance for processed data
- Shape validation: exact match required

## Adding New Datasets

To add validation for a new dataset (e.g., FOV1):

1. Generate reference data:
   ```bash
   uv run python simple_reference_generator.py --data-folder tests/data/FOV1 --output-dir reference_data
   ```

2. Update `conftest.py` to include the new dataset:
   ```python
   @pytest.fixture(params=["FOV18x", "FOV1"])
   def dataset_name(request):
       return request.param
   ```

3. Tests will automatically run against all datasets
