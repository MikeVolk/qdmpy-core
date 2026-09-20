"""Physical and algorithm constants for QDMpy.

Convention: All frequency values are in GHz.

This module defines physical constants and algorithm tuning parameters used
throughout the QDMpy package.  Algorithm tuning parameters are also accessible
from ``QDMpy.fitting.guess`` where they are primarily used.
"""

from __future__ import annotations

# Physical constants for NV centers
# Ref: Loubser & van Wyk, Rep. Prog. Phys. 41, 1201 (1978)
GAMMA_NV = 28.024  # GHz/T — from g = 2.0028(3), gamma = g * mu_B / h
D_ZFS = 2.870  # GHz, zero-field splitting (room temperature)

# Ref: Acosta et al., Phys. Rev. Lett. 104, 070801 (2010)
D_ZFS_TEMP_COEFFICIENT = -74e-6  # GHz/K, dD/dT = -74.2(7) kHz/K

# Hyperfine splitting constants for nitrogen isotopes (in GHz)
# Ref: Felton et al., Phys. Rev. B 79, 075203 (2009)
AHYP_14N = 0.002158  # GHz — A_parallel for 14N (I=1, triplet splitting)
AHYP_15N = 0.0015  # GHz — half of A_parallel=3.03 MHz for 15N (I=1/2, doublet)

# Magnetic field conversion factors
TESLA_TO_GAUSS = 1e4  # Convert Tesla to Gauss
GAUSS_TO_TESLA = 1e-4  # Convert Gauss to Tesla
MICROTESLA_TO_TESLA = 1e-6  # Convert microTesla to Tesla

# Algorithm tuning parameters for width estimation
DEFAULT_VMIN = 0.3
DEFAULT_VMAX = 0.7

# Fluorescence correction threshold (normalized units)
FLUORESCENCE_DELTA_THRESHOLD = 0.001

# Coordinate system labels for ODMR data
POLARITY_LABELS = ["neg", "pos"]
FRANGE_LABELS = ["low", "high"]
