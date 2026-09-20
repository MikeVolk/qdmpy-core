# Models Module

The models module provides spectral models for fitting ODMR data, including:

- Base model classes
- Specific models for different NV center configurations (14N, 15N, single resonance)
- A model registry system

## Model (Base Class)

The `Model` class is the base class for all spectral models in qdmpy.

::: qdmpy.fitting.models.Model
    options:
      show_root_heading: true
      show_source: true

## ModelRegistry

The `ModelRegistry` class manages model registration and access.

::: qdmpy.fitting.models.ModelRegistry
    options:
      show_root_heading: true
      show_source: true

## Specific Models

### ESRSINGLE

Single resonance model for ODMR spectra.

::: qdmpy.fitting.models.ESRSINGLE
    options:
      show_root_heading: true
      show_source: true

### ESR14N

Model for 14N NV centers with three hyperfine-split resonances.

::: qdmpy.fitting.models.ESR14N
    options:
      show_root_heading: true
      show_source: true

### ESR15N

Model for 15N NV centers with two hyperfine-split resonances.

::: qdmpy.fitting.models.ESR15N
    options:
      show_root_heading: true
      show_source: true
