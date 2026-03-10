# ODMR Spectral Models Tutorial

This tutorial explains how to understand and use the ODMR spectral models in qdmpy for fitting nitrogen-vacancy (NV) center data.

## Overview of ODMR Models

qdmpy provides three built-in spectral models for fitting ODMR data from nitrogen-vacancy centers:

1. **ESR14N** - For NV centers with ¹⁴N isotope (3 dips)
2. **ESR15N** - For NV centers with ¹⁵N isotope (2 dips)
3. **ESRSINGLE** - For single resonance systems (1 dip)

These models are implemented using Lorentzian lineshapes and are optimized for GPU-accelerated fitting.

## Model Architecture

The model system in qdmpy consists of:

- **Model Functions**: Core mathematical functions (`esr14n`, `esr15n`, `esrsingle`)
- **Model Classes**: Object-oriented wrappers (`ESR14N`, `ESR15N`, `ESRSINGLE`)
- **ModelRegistry**: Central registry for model management and retrieval

## Understanding the Models

### ESR14N Model

The ESR14N model represents ODMR spectra from NV centers with ¹⁴N nitrogen isotope (nuclear spin I=1):

```python
from qdmpy.models import ModelRegistry
import numpy as np

# Get the 14N model
model_14n = ModelRegistry.get('ESR14N')
print(f"Model: {model_14n.name}")
print(f"Number of parameters: {model_14n.n_parameters}")
print(f"Number of peaks: {model_14n.n_peaks}")
print(f"Parameters: {model_14n.parameters_unique}")
```

**Physics**: The ¹⁴N nucleus has three nuclear spin states (mI = -1, 0, +1), creating three hyperfine-split resonance lines.

**Parameters** (6 total):
- `center`: Center frequency of the resonance (GHz)
- `width_0`, `width_1`, `width_2`: Linewidth for each of the three dips (GHz)
- `contrast`: Overall contrast level (0-1)
- `offset`: Baseline offset (0-1)

### ESR15N Model

The ESR15N model represents ODMR spectra from NV centers with ¹⁵N nitrogen isotope (nuclear spin I=1/2):

```python
# Get the 15N model
model_15n = ModelRegistry.get('ESR15N')
print(f"Model: {model_15n.name}")
print(f"Number of parameters: {model_15n.n_parameters}")
print(f"Number of peaks: {model_15n.n_peaks}")
```

**Physics**: The ¹⁵N nucleus has two nuclear spin states (mI = -1/2, +1/2), creating two hyperfine-split resonance lines.

**Parameters** (5 total):
- `center`: Center frequency of the resonance (GHz)
- `width_0`, `width_1`: Linewidth for each of the two dips (GHz)
- `contrast`: Overall contrast level (0-1)
- `offset`: Baseline offset (0-1)

### ESRSINGLE Model

The ESRSINGLE model represents systems with a single resonance dip:

```python
# Get the single resonance model
model_single = ModelRegistry.get('ESRSINGLE')
print(f"Model: {model_single.name}")
print(f"Number of parameters: {model_single.n_parameters}")
print(f"Number of peaks: {model_single.n_peaks}")
```

**Use Cases**:
- Isotopically pure samples without hyperfine interaction
- Broadened spectra where hyperfine structure is not resolved
- Individual component fitting of complex spectra
- Initial parameter estimation

**Parameters** (4 total):
- `center`: Center frequency of the resonance (GHz)
- `width_0`: Linewidth of the dip (GHz)
- `contrast`: Contrast level (0-1)
- `offset`: Baseline offset (0-1)

## Using Models for Evaluation

You can evaluate any model directly with parameters:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create frequency array in GHz
frequencies = np.linspace(2.87, 2.88, 1000)

# Example parameters for ESR14N
# [center, width, contrast_-1, contrast_0, contrast_+1, offset]
params_14n = np.array([2.875, 0.003, 0.1, 0.2, 0.1, 0.0])

# Evaluate the model
model_14n = ModelRegistry.get('ESR14N')
spectrum = model_14n.func(frequencies, params_14n)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(frequencies, spectrum[0])
plt.xlabel('Frequency (GHz)')
plt.ylabel('Normalized Fluorescence')
plt.title('ESR14N Model Response')
plt.grid(True, alpha=0.3)
plt.show()
```

## Model Selection Guidelines

### When to use ESR14N:
- Working with natural diamond (99% ¹⁴N isotope)
- Well-resolved hyperfine structure visible
- Need to fit all three hyperfine components independently

### When to use ESR15N:
- Working with isotopically enriched ¹⁵N diamond
- Two-peak structure is clearly visible
- Smaller hyperfine splitting than ¹⁴N

### When to use ESRSINGLE:
- Highly broadened spectra (e.g., due to strain or high temperature)
- Proof-of-concept measurements
- When hyperfine structure is not resolved
- Fitting individual components of complex multi-NV spectra

## Model Registry Operations

The ModelRegistry provides convenient access to all available models:

```python
from qdmpy.models import ModelRegistry

# List all available models
all_models = ModelRegistry.all()
print("Available models:")
for name, info in all_models.items():
    print(f"  {name}: {info['class'].__name__}")

# Get model information
for model_name in ['ESR14N', 'ESR15N', 'ESRSINGLE']:
    model = ModelRegistry.get(model_name)
    print(f"\n{model_name}:")
    print(f"  Parameters: {model.n_parameters}")
    print(f"  Peaks: {model.n_peaks}")
    print(f"  Hyperfine constant: {all_models[model_name]['hyp']} GHz")
```

## Advanced Usage

### Parameter Constraints

When using models with fitting routines, you can specify constraints:

```python
# Example constraint dictionary for ESR14N
constraints = {
    'center': [2.8, 2.9],          # Center frequency bounds (GHz)
    'width': [0.001, 0.01],        # Linewidth bounds (GHz)
    'contrast': [0.0, 1.0],        # Contrast bounds
    'offset': [-0.1, 0.1],         # Offset bounds
}

# Convert to constraint array for fitting
model = ModelRegistry.get('ESR14N')
constraint_array = model.get_constraint_array(constraints)
```

### Mathematical Formulation

All models implement Lorentzian absorption lines:

**Single Lorentzian**:
```
f(x) = 1 + offset - (contrast × width² / ((x - center)² + width²))
```

**Multi-peak models** sum multiple Lorentzians at hyperfine-shifted positions.

### Performance Considerations

- Models are optimized for GPU acceleration via pyGpufit
- Use appropriate model complexity for your data quality
- Start with ESRSINGLE for initial parameter estimation
- Use vectorized parameter arrays for batch processing

## Integration with qdmpy Workflow

Models integrate seamlessly with the qdmpy fitting and measurement infrastructure:

```python
from qdmpy import Measurement
from qdmpy.models import ModelRegistry

# In a typical workflow:
# 1. Load ODMR data
# 2. Select appropriate model
model = ModelRegistry.get('ESR14N')  # or 'ESR15N', 'ESRSINGLE'

# 3. The model is automatically used by fitting routines
# measurement.fit_odmr(model=model)
```

For complete examples including custom model registration, see [03 · Extending the Framework](03-extending.ipynb).

## Summary

qdmpy's model system provides:
- **Three robust models** covering common NV center configurations
- **Physics-based implementations** with proper hyperfine splitting
- **Flexible parameter management** with constraint support
- **GPU-optimized performance** for large-scale fitting

Choose the model that best matches your experimental system and data quality. The models are designed to work seamlessly with qdmpy's fitting infrastructure while providing the flexibility needed for diverse ODMR applications.
