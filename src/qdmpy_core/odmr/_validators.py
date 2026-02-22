"""Internal validation utilities for ODMR data.

Not part of the public API — import from submodule paths, not from here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qdmpy_core.exceptions import DataValidationError

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
