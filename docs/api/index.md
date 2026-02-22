# API Reference

Welcome to the qdmpy_core API Reference. This section provides detailed documentation for all modules, classes, and functions in the qdmpy_core package.

## Package Structure

qdmpy_core is organized into several main modules:

- **Core**: Fundamental classes for data management and analysis
  - `fit`: Parameter fitting with constraint management
  - `models`: Spectral models for ODMR data analysis
  - `measurement`: Base classes for scientific measurements

- **ODMR**: Optically Detected Magnetic Resonance functionality
  - `odmr`: Main ODMR data structures and methods
  - `processors`: Signal processing algorithms for ODMR data
  - `data`: ODMR data management
  - `io`: Input/output operations for ODMR data

- **Utilities**:
  - `utils`: General utility functions
  - `constants`: Physical and mathematical constants
  - `exceptions`: Custom exception classes

## Using the API

The qdmpy_core API is designed to be intuitive and flexible. Most users will interact primarily with the high-level classes:

```python
import qdmpy_core

# Load ODMR data
odmr = qdmpy_core.ODMR.from_files(['data.mat'])

# Process the data
odmr.process_data()

# Fit the data
fit = qdmpy_core.FitManager(odmr.processed_data, odmr.frequencies)
fit.fit_odmr()

# Get results
centers = fit.get_param('center')
```

Refer to the specific module documentation for more details on each component.
