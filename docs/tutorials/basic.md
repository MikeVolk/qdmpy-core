# Basic qdmpy_core Tutorial

This tutorial introduces the fundamental concepts of qdmpy_core for analyzing ODMR data from NV centers in diamond.

[View the full tutorial notebook](../tutorial.ipynb)

## Quick Start

qdmpy_core provides a complete workflow for ODMR analysis:

1. **Load** data from various formats (.mat, .csv)
2. **Process** data with built-in processors  
3. **Fit** spectra using physics-based models
4. **Visualize** results with built-in plotting
5. **Export** processed data and results

## Loading Data

```python
from qdmpy_core.odmr.io import MatlabLoader
from qdmpy_core.odmr import ODMRData

# Load data from MATLAB files
loader = MatlabLoader(data_folder="./data")
raw_data, scan_dimensions, frequencies = loader.load()

# Create ODMRData object
odmr_data = ODMRData(raw_data, scan_dimensions, frequencies)
```

## Processing Data

qdmpy_core uses a modular processing pipeline:

```python
from qdmpy_core.odmr import ODMR
from qdmpy_core.odmr.processors import BinningProcessor, NormalizationProcessor

# Create ODMR manager
odmr = ODMR(odmr_data)

# Add processing steps
odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
odmr.processor_manager.add_processor(NormalizationProcessor(method='max'))

# Process the data
odmr.process_data()
```

## Automatic Model Selection & Fitting

```python
from qdmpy_core.guess import guess_n_peaks, guess_model
from qdmpy_core.fit import Fit

# Detect peaks and select model
n_peaks, doubt, _ = guess_n_peaks(odmr.processed_data.data)
model = guess_model(n_peaks)

# Fit the data
fit_obj = Fit(odmr.processed_data.data, odmr.processed_data.frequencies, model.name)
fit_obj.fit_odmr()
```

## Creating Measurements

```python
from qdmpy_core.measurement import Measurement

# Combine ODMR data with optical images
measurement = Measurement(
    odmr=odmr,
    light_image=light_image,
    laser_image=laser_image,
    pixel_spacing=4e-6
)
```

For the complete tutorial with detailed explanations and examples, see [the full Jupyter notebook](../tutorial.ipynb).