# noqa: N999
"""QDMpy: A Python package for Quantum Diamond Microscopy data analysis.

This package provides tools for processing, analyzing, and visualizing data from
Quantum Diamond Microscopy (QDM) experiments. It includes modules for loading data,
processing ODMR spectra, fitting spectral data to models, and creating visualizations.

The package is designed to be modular and extensible, allowing users to customize
the data processing pipeline to meet their specific experimental needs.
"""

from __future__ import annotations

__version__ = '0.1.0a'

from QDMpy.field_processing import (
    BaseFieldProcessor,
    BlankSubtractor,
    FieldProcessingPipeline,
    HotPixelFilter,
    QuadraticBackgroundSubtractor,
    UpwardContinuation,
)
from QDMpy.magnetic_map import MagneticMap
from QDMpy.settings import (
    NvSettings,
    get_settings,
    is_pygpufit_available,
    reset_settings,
)

# Configure logging on import to prevent stderr output in notebooks
_ = get_settings()

__all__ = [
    'BaseFieldProcessor',
    'BlankSubtractor',
    'FieldProcessingPipeline',
    'HotPixelFilter',
    'MagneticMap',
    'NvSettings',
    'QuadraticBackgroundSubtractor',
    'UpwardContinuation',
    'get_settings',
    'is_pygpufit_available',
    'reset_settings',
]
