"""Physical and computational constants for QDMpy.

This module defines essential constants used throughout the QDMpy package:

- Physical constants: Fundamental values from quantum physics relevant to NV centers
- Hyperfine constants: Specific values for different nitrogen isotopes (14N, 15N)
- Conversion factors: Units conversion for magnetic field calculations
- Default parameters: Standard values for algorithm parameters and thresholds
- System constants: Parameters related to experimental setup and hardware

These constants ensure consistency in calculations across the entire package
and provide physically accurate values for quantum diamond microscopy analysis.
"""

from __future__ import annotations

# Physical constants for NV centers
GAMMA_NV = 28.024e9  # Hz/T, NV center gyromagnetic ratio (more precise value from old QDM)
D_ZFS = 2.870e9  # Hz, zero-field splitting for NV centers

# Conversion constant from old QDM implementation
GAMMA = 28.024 / 1e6  # GHz/μT, gyromagnetic ratio in convenient units for B111 calculations

# Hyperfine splitting constants for nitrogen isotopes (in GHz)
AHYP_14N = 0.002158  # Hyperfine splitting constant for 14N (in GHz)
AHYP_15N = 0.0015  # Hyperfine splitting constant for 15N (in GHz)

# Magnetic field conversion factors
TESLA_TO_GAUSS = 1e4  # Convert Tesla to Gauss
GAUSS_TO_TESLA = 1e-4  # Convert Gauss to Tesla
MICROTESLA_TO_TESLA = 1e-6  # Convert microTesla to Tesla

# Default values for data processing algorithms
DEFAULT_VMIN = 0.3  # Default minimum value for normalized data in peak width estimation
DEFAULT_VMAX = 0.7  # Default maximum value for normalized data in peak width estimation
PROMINENCE = 0.0004  # Default prominence value for peak detection
