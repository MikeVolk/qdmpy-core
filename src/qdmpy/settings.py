"""Pydantic settings for QDMpy configuration.

This module defines the complete configuration schema for QDMpy using Pydantic,
supporting TOML files, environment variables, and programmatic overrides.
It also owns the application-level settings singleton via ``get_settings()``.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_PATH: Path = Path.home() / ".config" / "QDMpy"
CONFIG_FILE: Path = CONFIG_PATH / "settings.toml"


class ModelConstraintsSettings(BaseModel):
    """Settings for model fitting constraints.

    Supports two constraint specification modes controlled by ``constraint_units``:

    - ``'mt'`` (default): User specifies center/width bounds in millitesla (Zeeman
      shift). Converted internally to absolute GHz for the optimizer.
    - ``'absolute_ghz'``: User specifies center/width bounds directly in absolute
      GHz (power-user / backward-compatibility mode).

    In both modes the optimizer always receives absolute-GHz constraints.
    """

    constraint_units: Literal["mt", "absolute_ghz"] = Field(
        default="mt",
        description='Unit mode for center/width constraints: "mt" or "absolute_ghz"',
    )

    # -- mT mode (default) --
    center_max_mt: float = Field(default=1.1, description="Max Zeeman shift in mT (mt mode)")
    center_min_mt: float = Field(default=0.0, description="Min Zeeman shift in mT (mt mode)")
    width_max_mt: float = Field(default=0.08, description="Max linewidth in mT (mt mode)")
    width_min_mt: float = Field(default=0.017, description="Min linewidth in mT (mt mode)")

    # -- absolute GHz mode (power users / backward compat) --
    center_min: float = Field(default=2, description="Center frequency minimum (absolute GHz)")
    center_max: float = Field(default=3.1, description="Center frequency maximum (absolute GHz)")
    width_min: float = Field(default=0.0001, description="Width minimum (GHz)")
    width_max: float = Field(default=0.005, description="Width maximum (GHz)")

    # -- constraint types (shared, unitless) --
    center_type: Literal["FREE", "LOWER", "UPPER", "LOWER_UPPER"] = Field(
        default="LOWER_UPPER", description="Center constraint type"
    )
    width_type: Literal["FREE", "LOWER", "UPPER", "LOWER_UPPER"] = Field(
        default="LOWER_UPPER", description="Width constraint type"
    )
    contrast_min: float = Field(default=0.003, description="Contrast minimum")
    contrast_max: float = Field(default=0, description="Contrast maximum")
    contrast_type: Literal["FREE", "LOWER", "UPPER", "LOWER_UPPER"] = Field(
        default="LOWER", description="Contrast constraint type"
    )
    offset_min: float = Field(default=0, description="Offset minimum")
    offset_max: float = Field(default=0, description="Offset maximum")
    offset_type: Literal["FREE", "LOWER", "UPPER", "LOWER_UPPER"] = Field(
        default="FREE", description="Offset constraint type"
    )

    model_config = ConfigDict(extra="forbid")


class ModelSettings(BaseModel):
    """Settings for model configuration."""

    constraints: ModelConstraintsSettings = Field(
        default_factory=ModelConstraintsSettings, description="Fitting constraints"
    )

    model_config = ConfigDict(extra="forbid")


class FitSettings(BaseModel):
    """Settings for fitting."""

    estimator: Literal["LSE", "MLE"] = Field(default="MLE", description="Estimator type")
    max_number_iterations: int = Field(default=1000, description="Maximum iterations for fitting")
    tolerance: float = Field(default=1e-10, description="Fitting tolerance")
    backend: Literal["auto", "gpufit", "scipy", "torch"] = Field(
        default="auto",
        description=(
            "Fit optimizer backend. 'auto' uses gpufit if available, else "
            "torch when a real GPU device (cuda/mps) exists, else raises "
            "(never a silent CPU fallback). 'torch' (any device incl. CPU) "
            "and 'scipy' are explicit opt-ins. See qdmpy.fitting.backends."
        ),
    )

    model_config = ConfigDict(extra="forbid")


class LoggingSettings(BaseModel):
    """Settings for logging."""

    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    log_file: str | None = Field(
        default=None,
        description="Optional file path for persistent log output (supports rotation)",
    )
    enable_structured_logging: bool = Field(
        default=True, description="Enable structured JSON logging to file"
    )
    structured_log_dir: str | None = Field(
        default=None,
        description="Directory for structured JSON logs (defaults to ~/logs)",
    )

    model_config = ConfigDict(extra="forbid")


class NvSettings(BaseModel):
    """Settings for NV centre geometry.

    Used for B111 → Bxyz reconstruction; stores NV axis orientation in lab frame.
    """

    axis: tuple[float, float, float] = Field(
        default=(0.0, 0.8164966, 0.5773503),
        description="NV unit vector (ux, uy, uz) in lab frame. Default: QDM2 [111] orientation.",
    )
    epsilon: float = Field(
        default=1e-30,
        description="Regularisation term added to wavenumbers to avoid k=0 singularity.",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class QDMpySettings(BaseSettings):
    """Main QDMpy settings class.

    ``extra='forbid'`` throughout the tree: a typo in ``settings.toml``
    (``center_min_ml``) used to be dropped in silence, so the fit ran under a
    default constraint the user never chose and never found out. Unknown keys
    now raise at load time. Stray ``QDMPY_*`` environment variables are
    unaffected -- pydantic-settings only maps env vars onto declared fields.
    """

    model: ModelSettings = Field(default_factory=ModelSettings, description="Model settings")
    fit: FitSettings = Field(default_factory=FitSettings, description="Fitting settings")
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings, description="Logging settings"
    )
    nv: NvSettings = Field(default_factory=NvSettings, description="NV centre geometry settings")

    model_config = SettingsConfigDict(
        env_prefix="QDMPY_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls: type[QDMpySettings],
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        **kwargs: object,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources with TOML file support.

        Priority (highest to lowest):
        1. init_settings (constructor kwargs)
        2. env_settings (environment variables)
        3. TOML file (~/.config/QDMpy/settings.toml)
        4. Default values
        """
        toml_settings = (
            TomlConfigSettingsSource(settings_cls, CONFIG_FILE) if CONFIG_FILE.exists() else None
        )
        return tuple(s for s in [init_settings, env_settings, toml_settings] if s is not None)


# ---------------------------------------------------------------------------
# Settings singleton
# ---------------------------------------------------------------------------


def make_configfile(reset: bool = False) -> None:
    """Create the config directory; optionally delete the user TOML so defaults take over.

    Args:
        reset: If True, removes the user config file so Pydantic defaults take over.
    """
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    if reset and CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        logger.info("Deleted user config file {}", CONFIG_FILE)


def reset_config() -> None:
    """Delete the user config file and invalidate the cached settings."""
    make_configfile(reset=True)
    reset_settings()
    logger.info("Config reset to defaults")


def _configure_logging(settings: QDMpySettings) -> None:
    """Configure loguru with console and optional structured JSON sink."""
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)

    logger.remove()

    # Console sink: human-readable, no serialization
    logger.add(sys.stdout, level=settings.logging.log_level)

    # Structured JSON sink: DEBUG level, all context preserved
    if settings.logging.enable_structured_logging:
        log_dir = Path(settings.logging.structured_log_dir or Path.home() / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "qdmpy-{time:YYYY-MM-DD}.log"

        logger.add(
            str(log_file),
            level="DEBUG",
            format="{message}",
            serialize=True,
            rotation="10 MB",
            retention="7 days",
        )

    # Legacy file sink (backward compatibility)
    if settings.logging.log_file:
        logger.add(
            settings.logging.log_file,
            level=settings.logging.log_level,
            rotation="10 MB",
            retention="7 days",
        )


@lru_cache(maxsize=1)
def get_settings() -> QDMpySettings:
    """Return the lazily-initialised application settings singleton.

    The result is cached; call ``reset_settings()`` to force re-initialisation
    (e.g. after writing a new config file or in tests).
    """
    make_configfile()
    settings = QDMpySettings()
    _configure_logging(settings)
    return settings


def reset_settings() -> None:
    """Invalidate the cached settings so the next ``get_settings()`` re-reads config."""
    get_settings.cache_clear()


def is_pygpufit_available() -> bool:
    """Return True if the pygpufit GPU fitting library can be imported.

    Catches both ``ImportError`` (package not installed) and ``OSError``
    (package installed but its native library fails to load — e.g. a Linux
    wheel's ``.so`` on macOS/Windows) so an incompatible pyGpufit install
    degrades to "unavailable" instead of crashing at import time.
    """
    try:
        import pygpufit.gpufit  # noqa: F401
    except (ImportError, OSError):
        return False
    else:
        return True
