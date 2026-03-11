"""Visualization module for QDMpy.

This package provides plotting functions for visualizing data from Quantum Diamond
Microscopy (QDM) measurements, including magnetic field maps and spatial parameter maps.

All public symbols are re-exported here so that ``from qdmpy.plotting import <name>``
continues to work after the split into submodules.
"""

from __future__ import annotations

from qdmpy.plotting._common import resolve_pixel_indices
from qdmpy.plotting.display import plot_measurement_images, plot_qdm_display
from qdmpy.plotting.fields import plot_magnetic_component
from qdmpy.plotting.fit import (
    plot_b111_map,
    plot_fit_result_field_map,
    plot_fit_result_overview,
    plot_fit_result_parameter_map,
)
from qdmpy.plotting.odmr import (
    plot_fluorescence_correction,
    plot_folding_mean_spectrum,
    plot_folding_overview,
    plot_folding_pixel_spectra,
    plot_folding_search_landscape,
    plot_model_detection,
    plot_odmr_spectra,
)
from qdmpy.utils import double_norm

__all__ = [
    "double_norm",
    "plot_b111_map",
    "plot_fit_result_field_map",
    "plot_fit_result_overview",
    "plot_fit_result_parameter_map",
    "plot_fluorescence_correction",
    "plot_folding_mean_spectrum",
    "plot_folding_overview",
    "plot_folding_pixel_spectra",
    "plot_folding_search_landscape",
    "plot_magnetic_component",
    "plot_measurement_images",
    "plot_model_detection",
    "plot_odmr_spectra",
    "plot_qdm_display",
    "resolve_pixel_indices",
]
