# API Reference

Welcome to the QDMpy API Reference. This section provides detailed documentation for all modules, classes, and functions in the QDMpy package.

## Package Structure

QDMpy is organized into several main modules:

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

The QDMpy API is designed to be intuitive and flexible. Most users will interact primarily with the high-level classes:

```python
import QDMpy

# Load ODMR data
odmr = QDMpy.ODMR.from_files(['data.mat'])

# Process the data
odmr.process_data()

# Fit the data
fit = QDMpy.FitManager(odmr.processed_data, odmr.frequencies)
fit.fit_odmr()

# Get results
centers = fit.get_param('center')
```

Refer to the specific module documentation for more details on each component.