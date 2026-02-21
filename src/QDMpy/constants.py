"""Physical and algorithm constants for QDMpy.

Convention: All frequency values are in GHz.

This module defines physical constants and algorithm tuning parameters used
throughout the QDMpy package.  Algorithm tuning parameters are also accessible
from ``QDMpy.fitting.guess`` where they are primarily used.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from QDMpy.exceptions import DataValidationError

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
POLARITY_LABELS = ['neg', 'pos']
FRANGE_LABELS = ['low', 'high']

# Frequency range bounds for NV centers (in GHz)
NV_FREQ_MIN_GHZ = 2.0
NV_FREQ_MAX_GHZ = 3.5


def validate_frequencies(frequencies: NDArray) -> None:
    """Validate frequency array for ODMR fitting.

    Checks finiteness and monotonicity (errors). Checks NV diamond range
    (warning only, since non-standard experiments exist).

    Args:
        frequencies: 1D or 2D frequency array in GHz.

    Raises:
        DataValidationError: If frequencies contain non-finite values or
            are not monotonically increasing.
    """
    from loguru import logger

    frequencies = np.asarray(frequencies)

    if frequencies.size == 0:
        msg = 'Frequency array must not be empty'
        raise DataValidationError(msg)

    if not np.all(np.isfinite(frequencies)):
        msg = 'Frequency array contains non-finite values'
        raise DataValidationError(msg)

    if frequencies.ndim <= 1:
        rows = [frequencies]
    else:
        rows = [frequencies[i] for i in range(frequencies.shape[0])]

    for row in rows:
        if row.size > 1 and not np.all(np.diff(row) > 0):
            msg = 'Frequency array must be monotonically increasing'
            raise DataValidationError(msg)

    flat = frequencies.ravel()
    if flat.min() < NV_FREQ_MIN_GHZ or flat.max() > NV_FREQ_MAX_GHZ:
        logger.warning(
            f'Frequencies [{flat.min():.3f}, {flat.max():.3f}] GHz '
            f'are outside expected NV diamond range '
            f'[{NV_FREQ_MIN_GHZ}, {NV_FREQ_MAX_GHZ}] GHz'
        )
