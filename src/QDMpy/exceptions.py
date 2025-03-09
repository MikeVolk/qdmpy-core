"""Custom exceptions used throughout the QDMpy package.

This module defines custom exceptions that provide more specific error information
than standard Python exceptions, making it easier to understand and debug errors
specific to QDMpy operations.
"""
from __future__ import annotations


class CantImportError(Exception):
    """Exception raised when a required module or package cannot be imported.

    This exception is typically raised when the QDMpy package attempts to
    import a required dependency that is not available.
    """


class WrongFileNumber(Exception):
    """Exception raised when an incorrect number of files is provided.

    This exception is typically raised during data loading operations when
    the number of provided files doesn't match the expected number.
    """


class ModelGuessNotPossible(Exception):
    """Exception raised when automatic model selection fails.

    This exception is raised when QDMpy cannot automatically determine an
    appropriate model for the provided ODMR data, usually due to ambiguous
    spectral features or poor signal quality.
    """
