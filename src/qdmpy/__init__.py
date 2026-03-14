"""QDMpy: A Python package for Quantum Diamond Microscopy data analysis.

This package provides tools for processing, analyzing, and visualizing data from
Quantum Diamond Microscopy (QDM) experiments. It includes modules for loading data,
processing ODMR spectra, fitting spectral data to models, and creating visualizations.

The package is designed to be modular and extensible, allowing users to customize
the data processing pipeline to meet their specific experimental needs.
"""

from __future__ import annotations

from os import PathLike

__version__ = "0.1.0a"


def load(
    path: str | PathLike,
    *,
    bin_factor: int = 1,
    model: str = "auto",
    pixel_spacing: float = 4e-6,
    normalize: bool = True,
    fluorescence_correction: float | None = 0.2,
    output_directory: str | PathLike | None = None,
) -> Measurement:
    """Load ODMR data from a folder and return a ready-to-fit Measurement.

    One-line entry point for the common case: load data, apply standard
    processing, and return a Measurement ready for fit_odmr().

    Args:
        path: Folder containing MATLAB .mat files from the QDM microscope.
        bin_factor: Spatial binning factor (1 = no binning, 2 = 2×2 bins, …).
        model: ESR model name ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE').
        pixel_spacing: Physical pixel size in metres (default 4 µm).
        normalize: Apply max-normalisation to ODMR spectra (default True).
        fluorescence_correction: Fluorescence correction factor. Pass None to
            skip fluorescence correction. Default is 0.2.
        output_directory: Directory for saved outputs. Defaults to path/results.

    Returns:
        Measurement configured and ready for fit_odmr().

    Example:
        >>> import qdmpy
        >>> result = qdmpy.load('/data/FOV18x').fit_odmr()
        >>> result.b111_remanent
    """
    from qdmpy.measurement import Measurement as _Measurement

    return _Measurement.from_folder(
        path,
        bin_factor=bin_factor,
        model=model,
        pixel_spacing=pixel_spacing,
        normalize=normalize,
        fluorescence_correction=fluorescence_correction,
        output_directory=output_directory,
    )


# --- Entry points (User 1) ---
# --- Field sources ---
# --- Field processing ---
# --- Source fitting ---
from qdmpy.field_processing import (
    BaseFieldProcessor,
    BlankSubtractor,
    FieldProcessingPipeline,
    HotPixelFilter,
    QuadraticBackgroundSubtractor,
    UpwardContinuation,
)
from qdmpy.field_source import FieldSource, MagneticModel, MagneticSource, UpwardContinuedSource

# --- Fitting ---
from qdmpy.fitting import FitManager, FitResult, Model, ModelRegistry

# --- I/O ---
from qdmpy.io import load_npz, load_qdm, save_npz, save_qdm

# --- Magnetic reconstruction ---
from qdmpy.magnetic_map import FieldReconstructor, MagneticMap
from qdmpy.measurement import Measurement
from qdmpy.odmr.data import ODMRData

# --- Data loading ---
from qdmpy.odmr.io import MatlabLoader
from qdmpy.odmr.manager import ODMR

# --- Processing ---
from qdmpy.odmr.processors import (
    BinningProcessor,
    FluorescenceCorrectionProcessor,
    NormalizationProcessor,
    OutlierProcessor,
    Processor,
)
from qdmpy.result import QDMResult

# --- Settings ---
from qdmpy.settings import (
    NvSettings,
    get_settings,
    is_pygpufit_available,
    reset_settings,
)
from qdmpy.source_fitting import FitSourceResult, compute_field, fit_sources

# --- Testing / tutorial utilities ---
from qdmpy.testing import (
    make_synthetic_fit_result,
    make_synthetic_odmr_data,
    make_synthetic_qdm_result,
)

# Logging is configured lazily on the first get_settings() call,
# not at import time. This avoids filesystem side effects (e.g.
# creating ~/logs/) for users who only import types.

__all__ = [
    # Entry points
    "load",
    "Measurement",
    "QDMResult",
    # Field sources
    "FieldSource",
    "MagneticModel",
    "MagneticSource",
    "UpwardContinuedSource",
    # I/O
    "load_npz",
    "load_qdm",
    "save_npz",
    "save_qdm",
    # Data loading
    "MatlabLoader",
    "ODMRData",
    "ODMR",
    # Processing
    "BinningProcessor",
    "FluorescenceCorrectionProcessor",
    "NormalizationProcessor",
    "OutlierProcessor",
    "Processor",
    # Fitting
    "FitManager",
    "FitResult",
    "Model",
    "ModelRegistry",
    # Magnetic reconstruction
    "FieldReconstructor",
    "MagneticMap",
    # Settings
    "NvSettings",
    "get_settings",
    "is_pygpufit_available",
    "reset_settings",
    # Testing / tutorial utilities
    "make_synthetic_fit_result",
    "make_synthetic_odmr_data",
    "make_synthetic_qdm_result",
    # Source fitting
    "FitSourceResult",
    "compute_field",
    "fit_sources",
    # Field processing
    "BaseFieldProcessor",
    "BlankSubtractor",
    "FieldProcessingPipeline",
    "HotPixelFilter",
    "QuadraticBackgroundSubtractor",
    "UpwardContinuation",
]
