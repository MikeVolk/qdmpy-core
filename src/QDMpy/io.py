"""File input/output operations for QDM data analysis.

This module provides a unified interface for reading and writing various data types
used in Quantum Diamond Microscopy analysis. Key capabilities include:

- Image handling: Loading and saving images in multiple formats (TIFF, PNG, JPG)
- Metadata extraction: Reading acquisition parameters from file headers
- Format detection: Automatically determining appropriate loaders for input files
- Data export: Saving analyzed results in standardized formats
- Batch processing: Handling multiple files and directories
- Path resolution: Finding and validating data file locations

This module serves as the boundary between the file system and the data structures
used by the QDMpy analysis pipeline.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import matplotlib.image as mpimg
import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


def has_csv(lst: Sequence[str | bytes | os.PathLike[Any]]) -> bool:
    """Check if a list of files contains a CSV file.

    Args:
        lst: List of file paths to check.

    Returns:
        True if at least one file has a .csv extension, False otherwise.
    """
    return any('.csv' in str(s).lower() for s in lst)


def get_image_file(lst: Sequence[str | bytes | os.PathLike[Any]]) -> str:
    """Get the path to the first image file in the list.

    Prefers CSV files if available, otherwise looks for JPG files.

    Args:
        lst: List of file paths to search.

    Returns:
        Path to the first suitable image file.

    Raises:
        ValueError: If no suitable image files are found.
    """
    if has_csv(lst):
        filtered_lst = [s for s in lst if '.csv' in str(s).lower()]
    else:
        filtered_lst = [s for s in lst if '.jpg' in str(s).lower()]

    if not filtered_lst:
        raise ValueError('No suitable image files found in the list')

    return str(filtered_lst[0])


def get_image(
    folder: str | bytes | os.PathLike[Any],
    lst: Sequence[str | bytes | os.PathLike[Any]],
) -> NDArray:
    """Load an image from a file in the specified folder.

    Attempts to load a CSV file first, falling back to JPG if no CSV is available.

    Args:
        folder: Path to the folder containing the image files.
        lst: List of file names to search for image files.

    Returns:
        Image data as a numpy array.

    Raises:
        ValueError: If no suitable image files are found or if the image cannot be loaded.
    """
    folder_str = str(folder)

    try:
        image_file = get_image_file(lst)
        file_path = os.path.join(folder_str, image_file)

        if image_file.lower().endswith('.csv'):
            # Try different delimiters for CSV files
            try:
                img = np.loadtxt(file_path, delimiter=',')
            except ValueError:
                # Fall back to default delimiter (whitespace)
                img = np.loadtxt(file_path)
        else:  # Assume it's an image format matplotlib can read
            img = mpimg.imread(file_path)

        return np.array(img)
    except Exception as e:
        raise ValueError(f'Failed to load image: {e!s}')
