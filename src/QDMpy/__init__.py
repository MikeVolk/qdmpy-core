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

from functools import cache

from QDMpy.settings import get_settings, make_configfile, reset_config, reset_settings

__all__ = ['get_settings', 'is_pygpufit_available', 'make_configfile', 'reset_config', 'reset_settings']


@cache
def is_pygpufit_available() -> bool:
    """Return True if the pygpufit GPU fitting library can be imported."""
    try:
        import pygpufit.gpufit
    except ImportError:
        return False
    else:
        return True
