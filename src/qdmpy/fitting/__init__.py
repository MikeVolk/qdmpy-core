"""ODMR spectral fitting subpackage for QDMpy.

This subpackage bundles the four tightly-coupled modules that form the fitting
domain: model definitions, initial-parameter guessing, fitting management, and
result storage.

Public API:
    FitManager: Orchestrates GPU-accelerated fitting over all frequency ranges.
    ConstraintManager: Manages per-parameter fitting bounds.
    ParameterGuesser: Generates initial parameter estimates.
    CONSTRAINT_TYPES: Ordered list of constraint type strings.
    FitResult: Lightweight result container with B111 physics helpers.
    Model: Abstract base class for ODMR spectral models.
    ModelRegistry: Registry / factory for Model instances.
    ESR14N, ESR15N, ESRSINGLE: Concrete model implementations.
"""

from __future__ import annotations

from qdmpy.fitting.constraints import CONSTRAINT_TYPES, ConstraintManager
from qdmpy.fitting.guesser import ParameterGuesser
from qdmpy.fitting.manager import FitManager
from qdmpy.fitting.models import ESR14N, ESR15N, ESRSINGLE, Model, ModelRegistry
from qdmpy.fitting.result import FitResult

__all__ = [
    "CONSTRAINT_TYPES",
    "ESR14N",
    "ESR15N",
    "ESRSINGLE",
    "ConstraintManager",
    "FitManager",
    "FitResult",
    "Model",
    "ModelRegistry",
    "ParameterGuesser",
]
