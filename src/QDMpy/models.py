"""Backwards-compatibility shim — import from ``QDMpy.fitting.models`` instead."""
from QDMpy.fitting.models import (
    ESR14N,
    ESR15N,
    ESRSINGLE,
    Model,
    ModelRegistry,
    _main_demo,
    esr14n,
    esr15n,
    esrsingle,
)

__all__ = [
    "ESR14N",
    "ESR15N",
    "ESRSINGLE",
    "Model",
    "ModelRegistry",
    "_main_demo",
    "esr14n",
    "esr15n",
    "esrsingle",
]
