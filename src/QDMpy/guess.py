"""Backwards-compatibility shim — import from ``QDMpy.fitting.guess`` instead."""
from QDMpy.fitting.guess import (
    DEFAULT_VMAX,
    DEFAULT_VMIN,
    PROMINENCE,
    cumsum_center,
    cumsum_contrast,
    cumsum_width,
    get_model_by_peaks,
    guess_model,
    guess_n_peaks,
    normalize_pixel,
    validate_array,
)

__all__ = [
    "DEFAULT_VMAX",
    "DEFAULT_VMIN",
    "PROMINENCE",
    "cumsum_center",
    "cumsum_contrast",
    "cumsum_width",
    "get_model_by_peaks",
    "guess_model",
    "guess_n_peaks",
    "normalize_pixel",
    "validate_array",
]
