# noqa: N999
"""QDMpy: A Python package for Quantum Diamond Microscopy data analysis.

This package provides tools for processing, analyzing, and visualizing data from
Quantum Diamond Microscopy (QDM) experiments. It includes modules for loading data,
processing ODMR spectra, fitting spectral data to models, and creating visualizations.

The package is designed to be modular and extensible, allowing users to customize
the data processing pipeline to meet their specific experimental needs.
"""

from __future__ import annotations

__version__ = "0.1.0a"

import logging
import os
import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.rcParams["figure.facecolor"] = "white"

PROJECT_PATH = Path(os.path.abspath(__file__)).parent
CONFIG_PATH = Path().home() / ".config" / "QDMpy"
CONFIG_FILE = CONFIG_PATH / "settings.toml"
DESKTOP = Path().home() / "Desktop"

if TYPE_CHECKING:
    from QDMpy.settings import QDMpySettings


############################### configfile stuff ######################################
def make_configfile(reset: bool = False) -> None:
    """Creates the config directory if it does not exist.

    Args:
      reset: If True, removes the user config file so Pydantic defaults take over.

    """
    from loguru import logger

    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    if reset and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        logger.info(f"Deleted user config file {CONFIG_FILE}")


def reset_config() -> None:
    """Resets the config to default settings."""
    from loguru import logger

    make_configfile(reset=True)
    logger.info("Config reset to defaults")


def _configure_logging(settings: QDMpySettings) -> None:
    """Configure loguru and suppress noisy third-party loggers."""
    from loguru import logger

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)

    logger.remove()
    logger.add(sys.stderr, level=settings.logging.log_level)


_settings: QDMpySettings | None = None


def get_settings() -> QDMpySettings:
    """Return the lazily-initialised application settings singleton."""
    global _settings  # noqa: PLW0603 — module singleton pattern
    if _settings is None:
        from QDMpy.settings import QDMpySettings

        make_configfile()
        _settings = QDMpySettings()
        _configure_logging(_settings)
    return _settings


def reset_settings() -> None:
    """Clear the cached settings so the next ``get_settings()`` re-reads config."""
    global _settings  # noqa: PLW0603 — module singleton pattern
    _settings = None


@cache
def is_pygpufit_available() -> bool:
    """Return True if the pygpufit GPU fitting library can be imported."""
    try:
        import pygpufit.gpufit
    except ImportError:
        return False
    else:
        return True


# Import important modules
from . import io


def test_data_location() -> Path:
    """Returns the default path to test data.

    This function provides a suggested location for test data. Users should
    override this by setting the QDMPY_TEST_DATA environment variable or
    by explicitly providing data paths to their functions.

    Returns:
        Path to the test data directory. Defaults to ~/QDMpy_test_data if
        QDMPY_TEST_DATA environment variable is not set.

    Note:
        This function no longer contains hardcoded system-specific paths.
        Set the QDMPY_TEST_DATA environment variable to specify your test data location.
    """
    test_data_env = os.environ.get("QDMPY_TEST_DATA")
    if test_data_env:
        return Path(test_data_env)

    # Default to a directory in the user's home folder
    return Path.home() / "QDMpy_test_data"
