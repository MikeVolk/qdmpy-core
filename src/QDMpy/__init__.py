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
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import tomli
from loguru import logger

mpl.rcParams["figure.facecolor"] = "white"

PROJECT_PATH = Path(os.path.abspath(__file__)).parent
CONFIG_PATH = Path().home() / ".config" / "QDMpy"
CONFIG_FILE = CONFIG_PATH / "config.ini"
CONFIG_INI = PROJECT_PATH / "config.ini"
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
    """Creates the config file if it does not exist.

    Args:
      reset: bool:  (Default value = False)

    """
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists() or reset:
        logger.info(f"Copying default QDMpy 'config.ini' file to {CONFIG_FILE}")
        shutil.copy2(CONFIG_INI, CONFIG_FILE)


def load_config(file: Path | str = CONFIG_FILE) -> dict:
    """Loads the config file.

    Args:
        file: Path to the config file. Defaults to the standard config file location.

    Returns:
        Dictionary with the config file contents.
    """
    logger.info(f"Loading config file: {file}")
    with open(file, "rb") as file_obj:
        return tomli.load(file_obj)


def reset_config() -> None:
    """Resets the config file to default settings.

    This function overwrites the existing config file with the default settings
    from the package's internal config.ini file.
    """
    make_configfile(reset=True)
    logger.info("Config file reset")


make_configfile()
SETTINGS = load_config()

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
