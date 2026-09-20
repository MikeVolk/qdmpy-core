# ODMR Module Tests

This directory contains pytest tests for the ODMR module in QDMpy.

## Tests Structure

- `test_data.py`: Tests for the `ODMRData` class and its functionality
- `test_io.py`: Tests for the data loading functionality
- `test_processors.py`: Tests for the data processing classes
- `test_odmr.py`: Tests for the main ODMR class that orchestrates the module

## Running Tests

To run these tests, use pytest:

```bash
# Run all ODMR tests
pytest tests/odmr

# Run with verbosity
pytest tests/odmr -v

# Run a specific test file
pytest tests/odmr/test_data.py

# Run a specific test class
pytest tests/odmr/test_odmr.py::TestODMR

# Run a specific test method
pytest tests/odmr/test_odmr.py::TestODMR::test_init_empty
```

## Test Coverage

These tests cover:

1. Data handling and storage in `ODMRData`
2. MATLAB file loading functionality in `MatlabLoader`
3. Data processing operations:
   - Normalization
   - Binning
   - Outlier detection
4. Processing pipeline management
5. Main ODMR class functionality:
   - Initialization
   - Data loading
   - Data processing
   - Reset functionality
   - Method chaining support
