# Custom Models Tutorial

This tutorial explains how to create and use custom spectral models in QDMpy.

[View the full tutorial notebook](../tutorial_models.ipynb)

## Model Architecture in QDMpy

QDMpy provides a flexible framework for defining spectral models:

1. The `Model` base class defines the interface for all models
2. The `ModelRegistry` manages model registration and retrieval
3. Concrete model classes (like `ESR14N`, `ESR15N`, `ESRSINGLE`) implement specific spectral shapes

## Creating a Custom Model

You can create a custom model by subclassing `Model` and implementing its abstract methods:

```python
import numpy as np
from QDMpy.models import Model, ModelRegistry

class DoubleLorentzian(Model):
    """Model with two Lorentzian dips of equal contrast."""
    
    def __init__(self):
        """Initialize the double Lorentzian model."""
        super().__init__(name='DOUBLELORENTZIAN')
        self.parameter = ['contrast', 'center_1', 'center_2', 'width', 'offset']
        self.parameters_unique = ['contrast', 'center_1', 'center_2', 'width_0', 'offset']
        self.n_parameters = len(self.parameter)
        self.n_peaks = 2
        self.model_id = 5  # Choose a unique ID not used by existing models
    
    def evaluate(self, params, x):
        """Evaluate the model with the given parameters at the given x values."""
        # Unpack parameters
        contrast = params[0]
        center_1 = params[1]
        center_2 = params[2]
        width = params[3]
        offset = params[4]
        
        # Calculate the spectrum
        spectrum = offset
        spectrum -= contrast * (width**2 / ((x - center_1)**2 + width**2))
        spectrum -= contrast * (width**2 / ((x - center_2)**2 + width**2))
        
        return spectrum

# Register the model
ModelRegistry.register('DOUBLELORENTZIAN', {'class': DoubleLorentzian})
```

## Using a Custom Model

Once registered, you can use your custom model like any built-in model:

```python
import matplotlib.pyplot as plt

# Get the registered model
double_lorentzian = ModelRegistry.get('DOUBLELORENTZIAN')

# Create some test data
frequencies = np.linspace(2.85e9, 2.90e9, 1000)
params = np.array([0.1, 2.87e9, 2.88e9, 3e6, 1.0])

# Evaluate the model
spectrum = double_lorentzian.evaluate(params, frequencies)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(frequencies/1e9, spectrum)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Signal (a.u.)')
plt.title('Double Lorentzian Model')
plt.grid(True, alpha=0.3)
plt.show()

# Use with FitManager
from QDMpy.fit import FitManager

fit_manager = FitManager(data, frequencies, model_name='DOUBLELORENTZIAN')
fit_manager.fit_odmr()
```

## Advanced Model Features

### Parameter Constraints

You can define custom constraints for your model:

```python
# Set constraints specific to your model
fit_manager.set_constraints('center_1', vmin=2.86e9, vmax=2.88e9, constraint_type='LOWER_UPPER')
fit_manager.set_constraints('center_2', vmin=2.87e9, vmax=2.89e9, constraint_type='LOWER_UPPER')
fit_manager.set_constraints('width_0', vmin=1e6, constraint_type='LOWER')
```

### Initial Parameter Estimation

For better fitting results, you can implement custom parameter estimation:

```python
def guess_double_lorentzian_params(data, frequencies):
    """Estimate initial parameters for double Lorentzian model."""
    # Find the two largest dips in the spectrum
    mean_spectrum = np.mean(data, axis=(0, 1, 3))
    smoothed = np.convolve(mean_spectrum, np.ones(5)/5, mode='same')
    baseline = np.percentile(smoothed, 90)
    
    # Find dips as points below a threshold
    dips = smoothed < (baseline - 0.05)
    dip_indices = np.where(dips)[0]
    
    # Group adjacent indices into dip regions
    regions = []
    current_region = []
    for i in dip_indices:
        if not current_region or i == current_region[-1] + 1:
            current_region.append(i)
        else:
            regions.append(current_region)
            current_region = [i]
    if current_region:
        regions.append(current_region)
    
    # Get center of each region (up to 2)
    centers = []
    for region in regions[:2]:
        min_idx = region[np.argmin(smoothed[region])]
        centers.append(frequencies[min_idx])
    
    # If we found fewer than 2 dips, estimate the second one
    while len(centers) < 2:
        centers.append(centers[0] + 10e6)  # 10 MHz away
    
    # Estimate other parameters
    contrast = baseline - np.min(smoothed)
    width = 5e6  # 5 MHz initial guess
    offset = baseline
    
    return [contrast, centers[0], centers[1], width, offset]
```

For the full tutorial with detailed explanations and examples, please see [the complete Jupyter notebook](../tutorial_models.ipynb).