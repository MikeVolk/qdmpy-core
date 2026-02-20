"""Domain-specific exceptions for QDMpy.

Hierarchy:
    QDMpyError
    +-- DataError
    |   +-- DataLoadError
    |   +-- DataNotLoadedError
    |   +-- DataValidationError
    |       +-- DataShapeError
    +-- FittingError
    |   +-- FitNotPerformedError
    |   +-- FitConvergenceError
    |   +-- ModelNotFoundError
    |   +-- ModelGuessNotPossibleError
    |   +-- ParameterError
    +-- ConfigurationError
    +-- DependencyError
"""

from __future__ import annotations


class QDMpyError(Exception):
    """Base exception for all QDMpy errors."""


# --- Data Errors ---


class DataError(QDMpyError):
    """Base for data-related errors."""


class DataLoadError(DataError):
    """Failed to load data from file or source."""


class DataNotLoadedError(DataError):
    """Data has not been loaded or is not available."""


class DataValidationError(DataError):
    """Data failed validation checks."""


class DataShapeError(DataValidationError):
    """Data array has unexpected shape or dimensions."""


# --- Fitting Errors ---


class FittingError(QDMpyError):
    """Base for fitting-related errors."""


class FitNotPerformedError(FittingError):
    """Attempted to access fit results before fitting."""


class FitConvergenceError(FittingError):
    """Fit did not converge within allowed iterations."""


class ModelNotFoundError(FittingError):
    """Requested model is not registered."""


class ModelGuessNotPossibleError(FittingError):
    """Cannot determine appropriate model from data."""


class ParameterError(FittingError):
    """Invalid or unknown fitting parameter."""


# --- Configuration Errors ---


class ConfigurationError(QDMpyError):
    """Invalid or missing configuration."""


# --- Dependency Errors ---


class DependencyError(QDMpyError):
    """Required dependency is not available."""

