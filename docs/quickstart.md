# Quick Start Guide

This guide will help you get started with qdmpy_core quickly for analyzing ODMR data from NV centers in diamond.

## Installation

Install qdmpy_core using pip:

```bash
pip install qdmpy_core
```

Or with UV (recommended):

```bash
uv pip install qdmpy_core
```

## Basic Usage

Here's a minimal example to get you started with qdmpy_core:

```python
import numpy as np
import matplotlib.pyplot as plt
import qdmpy_core

# Load ODMR data
odmr = qdmpy_core.ODMR.from_files(['path/to/your/data.mat'])

# Process the data
odmr.process_data()

# Fit the data
fit = qdmpy_core.FitManager(odmr.processed_data, odmr.frequencies)
fit.fit_odmr()

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(odmr.f_ghz, odmr.mean_odmr, 'o-', alpha=0.5, label='Data')

# Get fitted parameters
center = fit.get_param('center')
print(f"Resonance frequency: {np.mean(center)/1e9:.6f} GHz")

# Plot the center frequency map
plt.figure(figsize=(8, 8))
plt.imshow(center.reshape(odmr.img_shape), cmap='viridis')
plt.colorbar(label='Resonance Frequency (Hz)')
plt.title('Center Frequency Map')
plt.show()
```

## Next Steps

- Check out the [full tutorial](tutorials/basic.md) for more details
- Learn about [ODMR processors](tutorials/processors.md) for advanced data processing
- Explore [model fitting](tutorials/fitting.md) with the ConstraintManager
- Learn how to create [custom models](tutorials/models.md) for your specific needs

## Command Line Interface

qdmpy_core also provides a command-line interface for quick data analysis:

```bash
qdmpy process data.mat --output results.mat
```

For more details on the CLI, see the [CLI documentation](cli.md).