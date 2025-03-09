"""
Utility functions for the QDMpy package.

This module provides various utility functions used throughout the QDMpy package,
including data conversion, coordinate transformations, and file handling utilities.
"""

import logging
import os
import sys
from typing import Union, Tuple, Optional, Sequence, Any, List

import matplotlib.image as mpimg
import numpy as np
from numpy.typing import ArrayLike, NDArray

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LOG = logging.getLogger(__name__)

# Unit prefixes for millify function
MILLNAMES = ["n", "μ", "m", "", " K", " M", " B", " T"]


def millify(n: float, sign: int = 1) -> str:
    """Convert a number to a human readable string with appropriate unit prefix.

    Args:
        n: Number to convert.
        sign: Number of digits after the decimal point. Default is 1.

    Returns:
        Human readable string with appropriate unit prefix (e.g., "1.5K" for 1500).
    """
    # Calculate the appropriate unit index
    millidx = max(
        0, min(len(MILLNAMES) - 1, int(np.floor(0 if n == 0 else np.log10(abs(n)) / 3)))
    )

    return f"{n / 10 ** (3 * millidx):.{sign}f}{MILLNAMES[millidx + 3]}"


def idx2rc(idx: ArrayLike, shape: Tuple[int, ...]) -> Tuple[NDArray, NDArray]:
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


def rc2idx(rc: ArrayLike, shape: Tuple[int, ...]) -> NDArray:
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
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    kx: int = 3,
    ky: int = 3,
    order: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def has_csv(lst: Sequence[Union[str, bytes, os.PathLike[Any]]]) -> bool:
    """Check if a list of files contains a CSV file.

    Args:
        lst: List of file paths to check.

    Returns:
        True if at least one file has a .csv extension, False otherwise.
    """
    return any(".csv" in str(s).lower() for s in lst)


def get_image_file(lst: Sequence[Union[str, bytes, os.PathLike[Any]]]) -> str:
    """Get the path to the first image file in the list.
    
    Prefers CSV files if available, otherwise looks for JPG files.

    Args:
        lst: List of file paths to search.

    Returns:
        Path to the first suitable image file.
        
    Raises:
        IndexError: If no suitable image files are found.
    """
    if has_csv(lst):
        filtered_lst = [s for s in lst if ".csv" in str(s).lower()]
    else:
        filtered_lst = [s for s in lst if ".jpg" in str(s).lower()]
    
    if not filtered_lst:
        raise IndexError("No suitable image files found in the list")
        
    return str(filtered_lst[0])


def get_image(
    folder: Union[str, bytes, os.PathLike[Any]],
    lst: Sequence[Union[str, bytes, os.PathLike[Any]]],
) -> NDArray:
    """Load an image from a file in the specified folder.

    Attempts to load a CSV file first, falling back to JPG if no CSV is available.

    Args:
        folder: Path to the folder containing the image files.
        lst: List of file names to search for image files.

    Returns:
        Image data as a numpy array.
        
    Raises:
        IndexError: If no suitable image files are found.
        ValueError: If the image file cannot be loaded.
    """
    folder_str = str(folder)
    
    try:
        image_file = get_image_file(lst)
        file_path = os.path.join(folder_str, image_file)
        
        if image_file.lower().endswith('.csv'):
            img = np.loadtxt(file_path)
        else:  # Assume it's an image format matplotlib can read
            img = mpimg.imread(file_path)
            
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def double_norm(data: NDArray, axis: Optional[int] = None) -> NDArray:
    """Normalize data to range [0, 1] by subtracting minimum and dividing by maximum.

    Args:
        data: Array to normalize.
        axis: Axis along which to perform normalization. If None, normalize globally.

    Returns:
        Normalized data array with values in the range [0, 1].
    """
    # Create a copy to avoid modifying the input array
    result = data.copy()
    
    # Calculate min along the specified axis and expand dimensions to match data
    mn = np.expand_dims(np.min(result, axis=axis), data.ndim - 1)
    result -= mn
    
    # Calculate max along the specified axis and expand dimensions
    mx = np.expand_dims(np.max(result, axis=axis), data.ndim - 1)
    
    # Avoid division by zero
    mx = np.where(mx == 0, 1.0, mx)
    result /= mx
    
    return result


def main() -> None:
    """
    Main function for demonstration purposes.
    
    This shows examples of using the utility functions in this module.
    """
    # Example of millify function
    print("Examples of millify:")
    print(f"0.001 -> {millify(0.001, 2)}")
    print(f"1500 -> {millify(1500, 1)}")
    print(f"1.2e6 -> {millify(1.2e6, 1)}")
    
    # Example of double_norm
    data = np.array([1, 2, 5, 10])
    print(f"\nOriginal data: {data}")
    print(f"Normalized data: {double_norm(data)}")


if __name__ == "__main__":
    main()
