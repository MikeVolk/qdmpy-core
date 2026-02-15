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
from pathlib import Path

import matplotlib as mpl
from loguru import logger

mpl.rcParams["figure.facecolor"] = "white"

PROJECT_PATH = Path(os.path.abspath(__file__)).parent
CONFIG_PATH = Path().home() / ".config" / "QDMpy"
CONFIG_FILE = CONFIG_PATH / "settings.toml"
DESKTOP = Path().home() / "Desktop"

SRC_PATH = PROJECT_PATH.parent
sys.path.append(str(SRC_PATH))

### LOGGING ###
# Suppress noisy third-party loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)

logger.info("WELCOME TO QDMpy")
logger.debug(f"QDMpy version {__version__} installed at {PROJECT_PATH}")
logger.debug(f"QDMpy config file {CONFIG_FILE}")


############################### configfile stuff ######################################
def make_configfile(reset: bool = False) -> None:
    """Creates the config directory if it does not exist.

    Args:
      reset: If True, removes the user config file so Pydantic defaults take over.

    """
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    if reset and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        logger.info(f"Deleted user config file {CONFIG_FILE}")


def reset_config() -> None:
    """Resets the config to default settings.

    This removes the user config file, causing Pydantic settings to use defaults.
    """
    make_configfile(reset=True)
    logger.info("Config reset to defaults")


# Import settings before any other QDMpy modules
from QDMpy.settings import QDMpySettings

make_configfile()
SETTINGS: QDMpySettings = QDMpySettings()

# Wire loguru logging to the configured level
logger.remove()
logger.add(sys.stderr, level=SETTINGS.logging.log_level)

############################### CHECK IF pygpufit IS INSTALLED ###############################
import importlib.util

package = "pygpufit"
# find_spec will look for the package
PYGPUFIT_PRESENT = importlib.util.find_spec(package) is not None

if PYGPUFIT_PRESENT is None or sys.platform == "darwin":
    logger.error(
        "Can't import pyGpufit. The package is necessary for most of the calculations. "
        "Functionality of QDMpy will be greatly diminished."
    )
    wheel_path = os.path.join(SRC_PATH, "pyGpufit", "win", "pyGpufit-1.2.0-py2.py3-none-any.whl")
    logger.error(
        f"try running:\n>>> pip install --no-index --find-links={wheel_path} pyGpufit"
    )
else:
    import pygpufit.gpufit as gf

    logger.info(f"CUDA available: {gf.cuda_available()}")
    runtime, driver = gf.get_cuda_version()
    logger.info(f"CUDA versions runtime: {runtime}, driver: {driver}")


# Import important modules
from . import io

if __name__ == "__main__":
    logger.info("This is a module. It is not meant to be run as a script.")
    sys.exit(0)


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
