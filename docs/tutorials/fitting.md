# Model Fitting Tutorial

This tutorial explains how to use qdmpy_core's fitting capabilities, with a focus on the FitManager and ConstraintManager.

[View the full tutorial notebook](../tutorial_fitting.ipynb)

## ODMR Spectral Models

qdmpy_core includes several spectral models for fitting ODMR data:

- **ESRSINGLE**: A single Lorentzian dip for simple ODMR spectra
- **ESR14N**: Three Lorentzian dips for nitrogen-14 NV centers (hyperfine splitting)
- **ESR15N**: Two Lorentzian dips for nitrogen-15 NV centers (hyperfine splitting)

```python
import qdmpy_core
from qdmpy_core.models import ModelRegistry

# List all available models
available_models = ModelRegistry.all()
print(f"Available models: {list(available_models.keys())}")

# Look at details of a specific model
esrsingle = ModelRegistry.get("ESRSINGLE")
print(f"\nESRSINGLE model details:")
print(f"  Parameters: {esrsingle.parameter}")
print(f"  Unique parameters: {esrsingle.parameters_unique}")
print(f"  Number of parameters: {esrsingle.n_parameters}")
```

## Using the FitManager

The `FitManager` class handles all aspects of fitting ODMR spectra:

```python
import numpy as np
from qdmpy_core.fit import FitManager

# Create or load your ODMR data
# ...

# Create a FitManager instance
fit_manager = FitManager(data, frequencies)

# Display information about the fit manager
print(fit_manager)
print(f"Model name: {fit_manager.model_name}")
print(f"Model parameters: {fit_manager.model_params}")

# Get initial parameter guesses
initial_params = fit_manager.get_initial_parameter()
print(f"Initial parameter shape: {initial_params.shape}")

# Perform fitting
fit_manager.fit_odmr()

# Access fitted parameters
fitted_params = fit_manager.parameter
centers = fit_manager.get_param('center')
widths = fit_manager.get_param('width_0')
contrasts = fit_manager.get_param('contrast')
```

## Working with the ConstraintManager

The `ConstraintManager` provides a robust way to set constraints on fit parameters:

```python
# View current constraints
constraints = fit_manager.constraints
print("Default constraints:")
for param, constraint in constraints.items():
    print(f"  {param}:")
    print(f"    Min: {constraint[0]}")
    print(f"    Max: {constraint[1]}")
    print(f"    Type: {constraint[2]}")

# Set more restrictive constraints on center frequency
fit_manager.set_constraints(
    'center',
    vmin=2.87e9,      # Lower bound (in Hz)
    vmax=2.88e9,      # Upper bound (in Hz)
    constraint_type='LOWER_UPPER'
)

# Set a minimum width to avoid unrealistically narrow features
fit_manager.set_constraints(
    'width_0',
    vmin=2e6,         # Lower bound (in Hz)
    constraint_type='LOWER'
)

# Set a maximum contrast to avoid over-fitting
fit_manager.set_constraints(
    'contrast',
    vmax=0.2,         # Upper bound
    constraint_type='UPPER'
)

# Removing all constraints
fit_manager.set_free_constraints()
```

## Constraint Types

qdmpy_core supports four constraint types:

1. **FREE**: No constraints (parameters can take any value)
2. **LOWER**: Only a lower bound is applied
3. **UPPER**: Only an upper bound is applied
4. **LOWER_UPPER**: Both lower and upper bounds are applied

For the full tutorial with detailed explanations and examples, please see [the complete Jupyter notebook](../tutorial_fitting.ipynb).
