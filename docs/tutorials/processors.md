# ODMR Processors Tutorial

This tutorial explores the various signal processing techniques available in qdmpy for ODMR data analysis.

[View the full tutorial notebook](../processor_tutorial.ipynb)

## Available Processors

qdmpy includes several ODMR processors:

1. **NormalizationProcessor**: Normalizes data to a common scale
2. **BinningProcessor**: Performs spatial binning to improve SNR
3. **OutlierProcessor**: Identifies and removes outlier pixels
4. **FluorescenceCorrectionProcessor**: Corrects for global fluorescence variations

## Using the Processor Pipeline

The `ODMRProcessorManager` provides a convenient way to create and execute a processing pipeline:

```python
import qdmpy
from qdmpy.odmr.processors import (
    ODMRProcessorManager,
    NormalizationProcessor,
    BinningProcessor,
    OutlierProcessor
)

# Create an ODMR instance with data
odmr = qdmpy.ODMR.from_files(['data.mat'])

# Create a processor manager
processor_manager = ODMRProcessorManager()

# Add processors to the pipeline
processor_manager.add_processor(NormalizationProcessor(method='max'))
processor_manager.add_processor(BinningProcessor(bin_factor=2))
processor_manager.add_processor(OutlierProcessor(threshold=3.0))

# Process the data
processed_data = processor_manager.process(odmr.raw_data)

# Update the ODMR instance with processed data
odmr.processed_data = processed_data
```

## Custom Processors

You can create custom processors by subclassing `BaseProcessor`:

```python
from qdmpy.odmr.processors import BaseProcessor
import numpy as np

class SmoothingProcessor(BaseProcessor):
    """Processor that applies Gaussian smoothing to ODMR spectra."""

    def __init__(self, sigma=1.0):
        """Initialize with smoothing parameter."""
        self.sigma = sigma

    def process(self, data):
        """Apply smoothing to each spectrum."""
        from scipy.ndimage import gaussian_filter1d

        # Get data dimensions
        n_pol, n_frange, n_freq, n_pixel = data.shape

        # Create output array
        result = np.zeros_like(data)

        # Apply smoothing to each spectrum
        for pol in range(n_pol):
            for frange in range(n_frange):
                for pixel in range(n_pixel):
                    spectrum = data[pol, frange, :, pixel]
                    result[pol, frange, :, pixel] = gaussian_filter1d(
                        spectrum, sigma=self.sigma
                    )

        return result
```

For the full tutorial with detailed explanations and examples, please see [the complete Jupyter notebook](../processor_tutorial.ipynb).
