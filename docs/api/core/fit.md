# Fit Module

The fit module provides tools for fitting ODMR spectral data, with a focus on parameter estimation, constraint management, and result access.

## FitManager

The `FitManager` class is the main interface for fitting ODMR spectra. It handles model selection, parameter estimation, constraint management, and fitting execution.

::: QDMpy.fit.FitManager
    options:
      show_root_heading: true
      show_source: true

## ConstraintManager

The `ConstraintManager` class manages constraints on fit parameters, providing a robust way to set limits and constraint types.

::: QDMpy.fit.ConstraintManager
    options:
      show_root_heading: true
      show_source: true

## Constants and Types

::: QDMpy.fit.CONSTRAINT_TYPES
::: QDMpy.fit.ESTIMATOR_ID
::: QDMpy.fit.UNITS