"""Physical and algorithm constants for QDMpy.

Convention: All frequency values are in GHz.

This module defines physical constants and algorithm tuning parameters used
throughout the QDMpy package.  Algorithm tuning parameters are also accessible
from ``QDMpy.fitting.guess`` where they are primarily used.
"""

from __future__ import annotations

# Physical constants for NV centers
GAMMA_NV = 28.024  # GHz/T, NV center gyromagnetic ratio
D_ZFS = 2.870  # GHz, zero-field splitting for NV centers

# Hyperfine splitting constants for nitrogen isotopes (in GHz)
AHYP_14N = 0.002158  # Hyperfine splitting constant for 14N (in GHz)
AHYP_15N = 0.0015  # Hyperfine splitting constant for 15N (in GHz)

# Magnetic field conversion factors
TESLA_TO_GAUSS = 1e4  # Convert Tesla to Gauss
GAUSS_TO_TESLA = 1e-4  # Convert Gauss to Tesla
MICROTESLA_TO_TESLA = 1e-6  # Convert microTesla to Tesla

# Algorithm tuning parameters for peak detection and width estimation
DEFAULT_VMIN = 0.3
DEFAULT_VMAX = 0.7
PROMINENCE = 0.0004

# Fluorescence correction threshold (normalized units)
FLUORESCENCE_DELTA_THRESHOLD = 0.001

# Coordinate system labels for ODMR data
POLARITY_LABELS = ["neg", "pos"]
FRANGE_LABELS = ["low", "high"]
