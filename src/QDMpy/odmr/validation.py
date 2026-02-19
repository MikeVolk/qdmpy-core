"""Backwards-compatibility shim — import from ``QDMpy.odmr.data`` instead."""
from QDMpy.odmr.data import NV_FREQ_MAX_GHZ, NV_FREQ_MIN_GHZ, validate_frequencies

__all__ = ["NV_FREQ_MAX_GHZ", "NV_FREQ_MIN_GHZ", "validate_frequencies"]
