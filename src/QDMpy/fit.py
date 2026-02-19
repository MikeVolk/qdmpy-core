"""Backwards-compatibility shim — import from ``QDMpy.fitting`` instead."""
from QDMpy.fitting.manager import (
    CONSTRAINT_TYPES,
    ESTIMATOR_ID,
    ConstraintManager,
    FitManager,
    ParameterGuesser,
)

__all__ = [
    "CONSTRAINT_TYPES",
    "ESTIMATOR_ID",
    "ConstraintManager",
    "FitManager",
    "ParameterGuesser",
]
