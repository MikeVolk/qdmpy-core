"""Utility functions for the QDMpy package.

This module provides essential utility functions used throughout the QDMpy package:

- Data processing: Functions for filtering, smoothing, and conditioning data
- Coordinate transformations: Converting between different spatial reference systems
- Path management: Handling file paths and ensuring correct package imports
- Mathematical operations: Polynomial fitting and surface interpolation
- Array manipulation: Reshaping and transforming multi-dimensional data arrays
- Validation: Input checking and validation utilities

These utilities serve as the underlying infrastructure for the higher-level
functionality provided by other modules in the package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Unit prefixes for millify function
MILLNAMES = ["n", "μ", "m", "", " K", " M", " B", " T"]


def millify(n: float, sign: int = 1) -> str:
    """Convert a number to a human readable string with appropriate unit prefix.

    Args:
        n: Number to convert.
        sign: Number of decimal places to include in the output. Default is 1.

    Returns:
        Human readable string with appropriate unit prefix (e.g., "1.5K" for 1500).
    """
    # Calculate the appropriate unit index. MILLNAMES is indexed as
    # MILLNAMES[millidx + 3], so millidx must be clamped to [-3, len(MILLNAMES) - 1 - 3]
    # rather than to [0, len(MILLNAMES) - 1].
    lo = -3
    hi = len(MILLNAMES) - 1 - 3
    millidx = max(
        lo,
        min(hi, int(np.floor(0 if n == 0 else np.log10(abs(n)) / 3))),
    )

    return f"{n / 10 ** (3 * millidx):.{sign}f}{MILLNAMES[millidx + 3]}"


def idx2rc(idx: ArrayLike, shape: tuple[int, ...]) -> tuple[NDArray, NDArray]:
    """Convert a linear index to row-column (yx) coordinates.

    Args:
        idx: Linear indices to convert. Can be a single integer or an array of integers.
        shape: Shape of the array (height, width) the indices refer to.

    Returns:
        Tuple of (row_indices, column_indices) corresponding to the input linear indices.
    """
    idx = np.atleast_1d(idx)
    idx = np.array(idx).astype(int)
    return np.unravel_index(idx, shape)  # type: ignore[return-value]


def rc2idx(rc: ArrayLike, shape: tuple[int, ...]) -> NDArray:
    """Convert row-column (yx) coordinates to linear indices.

    Args:
        rc: Array of coordinates, as a tuple or array of form [[row_indices], [column_indices]].
        shape: Shape of the array (height, width) the coordinates refer to.

    Returns:
        Array of linear indices corresponding to the input row-column coordinates.
    """
    rc = np.array(rc).astype(int)
    return np.ravel_multi_index(rc, shape)  # type: ignore[call-overload]


def polyfit2d(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    kx: int = 3,
    ky: int = 3,
    order: int | None = None,
) -> tuple[NDArray, NDArray, int, NDArray]:
    """Two dimensional polynomial fitting by least squares.

    Fits the functional form f(x,y) = z, performing a polynomial fit in two dimensions.

    Args:
        x: Array of x values for the fit.
        y: Array of y values for the fit.
        z: Array of z values to fit (dependent variable).
        kx: Polynomial order in x direction. Default is 3.
        ky: Polynomial order in y direction. Default is 3.
        order: If provided, only coefficients with i+j <= order are included.
               Default is None, which includes all terms up to kx, ky.

    Returns:
        Tuple containing:
            - solution: Array of polynomial coefficients
            - residuals: Sum of squared residuals of the fit
            - rank: Rank of the coefficient matrix
            - singular_values: Singular values of the coefficient matrix

    Notes:
        The resulting fit can be evaluated with:
        >>> np.polynomial.polynomial.polygrid2d(x, y, solution.reshape((kx+1, ky+1)))

    Reference:
        Inspired by: https://stackoverflow.com/questions/33964913
    """
    # Create grid coordinates
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Create coefficient array, up to x^kx, y^ky
    coeffs = np.ones(shape=(kx + 1, ky + 1))

    # Create solve array
    a = np.zeros((coeffs.size, x_mesh.size))

    # For each coefficient, produce array x^i * y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # Do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x_mesh)
        else:
            arr = coeffs[i, j] * x_mesh**i * y_mesh**j
        a[index] = arr.ravel()

    # Perform least squares fitting
    solution, res, rank, s = np.linalg.lstsq(a.T, np.ravel(z), rcond=None)
    return solution, res, rank, s


def rms(data: NDArray) -> float:
    """Calculate the root mean square of a data set.

    Args:
        data: Array of numerical values.

    Returns:
        Root mean square of the data.
    """
    return float(np.sqrt(np.mean(np.square(data))))


# Image-related functions have been moved to io.py


def double_norm(data: NDArray, axis: int | None = None) -> NDArray:
    """Normalize data to range [0, 1] by subtracting minimum and dividing by maximum.

    Args:
        data: Array to normalize.
        axis: Axis along which to perform normalization. If None, normalize globally.

    Returns:
        Normalized data array with values in the range [0, 1].
    """
    # Create a copy to avoid modifying the input array
    result = data.copy()

    # keepdims broadcasts back against the original axis positions (including
    # every axis when axis=None), unlike expand_dims at a fixed position
    mn = np.min(result, axis=axis, keepdims=True)
    result -= mn

    # Calculate max along the specified axis, keeping dims for broadcasting
    mx = np.max(result, axis=axis, keepdims=True)

    # Avoid division by zero
    mx = np.where(mx == 0, 1.0, mx)
    result /= mx

    return result
