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

# Hyperfine splitting constants for nitrogen isotopes (in GHz)
from __future__ import annotations

AHYP_14N = 0.002158  # Hyperfine splitting constant for 14N (in GHz)
AHYP_15N = 0.0015    # Hyperfine splitting constant for 15N (in GHz)

# Default values for data processing algorithms
DEFAULT_VMIN = 0.3   # Default minimum value for normalized data in peak width estimation
DEFAULT_VMAX = 0.7   # Default maximum value for normalized data in peak width estimation
PROMINENCE = 0.0004  # Default prominence value for peak detection
