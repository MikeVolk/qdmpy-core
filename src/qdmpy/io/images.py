"""File I/O utilities for image loading.

Provides general-purpose file loading and image handling utilities
used across the package.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.image as mpimg
import numpy as np
from loguru import logger

from qdmpy.exceptions import DataLoadError

if TYPE_CHECKING:
    from numpy.typing import NDArray


def has_csv(lst: Sequence[str | bytes | os.PathLike[Any]]) -> bool:
    """Check if a list of files contains a CSV file.

    Args:
        lst: List of file paths to check.

    Returns:
        True if at least one file has a .csv extension, False otherwise.
    """
    return any(".csv" in str(s).lower() for s in lst)


def get_image_file(lst: Sequence[str | bytes | os.PathLike[Any]]) -> str:
    """Get the path to the first image file in the list.

    Prefers CSV files if available, otherwise looks for JPG files.

    Args:
        lst: List of file paths to search.

    Returns:
        Path to the first suitable image file.

    Raises:
        DataLoadError: If no suitable image files are found.
    """
    if has_csv(lst):
        filtered_lst = [s for s in lst if ".csv" in str(s).lower()]
    else:
        filtered_lst = [s for s in lst if ".jpg" in str(s).lower()]

    if not filtered_lst:
        msg = "No suitable image files found in the list"
        raise DataLoadError(msg)

    selected = str(filtered_lst[0])
    logger.debug("Selected image file: {}", selected)
    return selected


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
        DataLoadError: If no suitable image files are found or if the image
            cannot be loaded.
    """
    folder_str = str(folder)

    try:
        image_file = get_image_file(lst)
        file_path = os.path.join(folder_str, image_file)
        logger.debug("Loading image from: {}", file_path)

        if image_file.lower().endswith(".csv"):
            try:
                img = np.loadtxt(file_path, delimiter=",")
            except ValueError:
                logger.debug("CSV comma delimiter failed, falling back to whitespace")
                img = np.loadtxt(file_path)
        else:
            img = mpimg.imread(file_path)
    except Exception as e:
        msg = f"Failed to load image: {e!s}"
        raise DataLoadError(msg) from e
    else:
        result = np.array(img)
        logger.info("Loaded image {} with shape {}", file_path, result.shape)
        return result


def load_metadata_toml(folder: str | bytes | os.PathLike[Any]) -> dict[str, Any]:
    """Load metadata.toml from folder if present, return empty dict otherwise.

    Args:
        folder: Path to the folder containing metadata.toml.

    Returns:
        Dictionary parsed from metadata.toml, or empty dict if file not found
        or invalid.
    """
    path = Path(str(folder)) / "metadata.toml"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        logger.warning("metadata.toml at {} is invalid TOML and was skipped: {}", path, e)
        return {}
    except OSError as e:
        logger.warning("Could not read metadata.toml at {}: {}", path, e)
        return {}
    else:
        logger.debug("Loaded metadata from {}", path)
        return data
