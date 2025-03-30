# Basic QDMpy Tutorial

This tutorial introduces the fundamental concepts of QDMpy for analyzing ODMR data from NV centers in diamond.

[View the full tutorial notebook](../tutorial.ipynb)

```{jupyter-execute}
import QDMpy
print(f"QDMpy version: {QDMpy.__version__}")
```

## Loading Data

QDMpy can load ODMR data from various file formats:

```python
# Load data from a .mat file
odmr = QDMpy.ODMR.from_files(['data.mat'])

# Print information about the loaded data
print(odmr)
```

## Processing Data

ODMR data typically requires processing steps such as normalization, binning, and outlier removal:

```python
# Process the data with default parameters
odmr.process_data()

# Configure specific processing steps
odmr.normalize_data(method='max')
odmr.bin_data(bin_factor=2)
odmr.remove_outliers(threshold=3.0)
```

## Visualizing Data

QDMpy provides various visualization options:

```python
import matplotlib.pyplot as plt

# Plot the mean ODMR spectrum
plt.figure(figsize=(10, 6))
plt.plot(odmr.f_ghz, odmr.mean_odmr)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Signal (a.u.)')
plt.title('Mean ODMR Spectrum')
plt.grid(True, alpha=0.3)
plt.show()

# Plot a spatial map of contrast values
plt.figure(figsize=(8, 8))
plt.imshow(odmr.contrast.reshape(odmr.img_shape), cmap='viridis')
plt.colorbar(label='Contrast (a.u.)')
plt.title('ODMR Contrast Map')
plt.show()
```

For the full tutorial with detailed explanations and examples, please see [the complete Jupyter notebook](../tutorial.ipynb).