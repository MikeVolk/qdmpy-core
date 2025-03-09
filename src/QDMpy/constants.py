"""
Constants used throughout the QDMpy package.

This module defines physical and computational constants that are used in various
parts of the package. These include hyperfine constants for different nitrogen
isotopes and default values for algorithm parameters.
"""

# Hyperfine splitting constants for nitrogen isotopes (in GHz)
AHYP_14N = 0.002158  # Hyperfine splitting constant for 14N (in GHz)
AHYP_15N = 0.0015    # Hyperfine splitting constant for 15N (in GHz)

# Default values for data processing algorithms
DEFAULT_VMIN = 0.3   # Default minimum value for normalized data in peak width estimation
DEFAULT_VMAX = 0.7   # Default maximum value for normalized data in peak width estimation
PROMINENCE = 0.0004  # Default prominence value for peak detection