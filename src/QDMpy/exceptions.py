"""Custom exceptions for error handling in QDMpy.

This module defines domain-specific exceptions that provide precise error information
for QDMpy operations. Using these custom exceptions instead of generic Python exceptions
offers several advantages:

- Context-specific error messages: Clearer indication of what went wrong
- Hierarchical organization: Exceptions grouped by functional domain
- Consistent error handling: Standard patterns for different error types
- Better debugging: More informative tracebacks with contextual information
- Enhanced error recovery: Specific exception types allow targeted exception handling

Each exception is documented with its purpose and typical usage scenarios.
"""

from __future__ import annotations


class CantImportError(Exception):
    """Exception raised when a required module or package cannot be imported.

    This exception is typically raised when the QDMpy package attempts to
    import a required dependency that is not available.
    """


class WrongFileNumberError(Exception):
    """Exception raised when an incorrect number of files is provided.

    This exception is typically raised during data loading operations when
    the number of provided files doesn't match the expected number.
    """


class ModelGuessNotPossibleError(Exception):
    """Exception raised when automatic model selection fails.

    This exception is raised when QDMpy cannot automatically determine an
    appropriate model for the provided ODMR data, usually due to ambiguous
    spectral features or poor signal quality.
    """
