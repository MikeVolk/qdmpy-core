"""Persistence helpers for MagneticMap edge concerns."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from os import PathLike

    from qdmpy.magnetic_map import MagneticMap


def save_magnetic_map(magnetic_map: MagneticMap, path: str | PathLike) -> None:
    """Save a MagneticMap to NetCDF."""
    path_obj = Path(path)
    magnetic_map.to_dataset().to_netcdf(path_obj)
    logger.info("MagneticMap saved to {}", path_obj)
