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
import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

CONFIG_PATH = Path().home() / ".config" / "QDMpy"
CONFIG_FILE = CONFIG_PATH / "settings.toml"

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

    if settings.logging.log_file:
        logger.add(
            settings.logging.log_file,
            level=settings.logging.log_level,
            rotation="10 MB",
            retention="7 days",
        )


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
