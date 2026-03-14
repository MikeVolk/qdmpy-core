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


class DefaultPathsSettings(BaseModel):
    """Settings for default paths."""

    data_path: str = Field(default="", description="Default data path")


class OdmrSettings(BaseModel):
    """Settings for ODMR processing."""

    norm_method: Literal["mean"] = Field(
        default="mean", description="Normalization method for ODMR data"
    )


class ModelFindPeaksSettings(BaseModel):
    """Settings for model peak finding."""

    prominence: float = Field(default=0.0004, description="Prominence threshold")


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

    model_config = ConfigDict(extra="ignore")


class ModelSettings(BaseModel):
    """Settings for model configuration."""

    find_peaks: ModelFindPeaksSettings = Field(
        default_factory=ModelFindPeaksSettings,
        description="Peak finding settings",
    )
    constraints: ModelConstraintsSettings = Field(
        default_factory=ModelConstraintsSettings, description="Fitting constraints"
    )

    model_config = ConfigDict(extra="ignore")


class FitSettings(BaseModel):
    """Settings for fitting."""

    estimator: Literal["LSE", "MLE"] = Field(default="MLE", description="Estimator type")
    max_number_iterations: int = Field(default=1000, description="Maximum iterations for fitting")
    tolerance: float = Field(default=1e-10, description="Fitting tolerance")

    model_config = ConfigDict(extra="ignore")


class StatisticsPercentileSettings(BaseModel):
    """Settings for StatisticsPercentile outlier detection."""

    chi2_percentile: list[float] = Field(
        default=[0, 99.9], description="Chi-squared percentile bounds"
    )
    width_percentile: list[float] = Field(default=[0, 99.0], description="Width percentile bounds")
    contrast_percentile: list[float] = Field(
        default=[1, 100], description="Contrast percentile bounds"
    )

    model_config = ConfigDict(extra="ignore")


class LocalOutlierFactorSettings(BaseModel):
    """Settings for LocalOutlierFactor outlier detection."""

    n_neighbors: int = Field(default=20, description="Number of neighbors")
    algorithm: str = Field(default="auto", description="Algorithm type")
    leaf_size: int = Field(default=30, description="Leaf size")
    metric: str = Field(default="minkowski", description="Distance metric")
    p: int = Field(default=2, description="Minkowski p parameter")
    contamination: str | float = Field(default="auto", description="Expected outlier fraction")

    model_config = ConfigDict(extra="ignore")


class OutlierDetectionSettings(BaseModel):
    """Settings for outlier detection."""

    method: Literal["LocalOutlierFactor", "StatisticsPercentile"] = Field(
        default="LocalOutlierFactor", description="Outlier detection method"
    )
    statistics_percentile: StatisticsPercentileSettings = Field(
        default_factory=StatisticsPercentileSettings,
        description="StatisticsPercentile settings",
    )
    local_outlier_factor: LocalOutlierFactorSettings = Field(
        default_factory=LocalOutlierFactorSettings,
        description="LocalOutlierFactor settings",
    )

    model_config = ConfigDict(extra="ignore")


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

    model_config = ConfigDict(extra="ignore")


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

    model_config = ConfigDict(frozen=True, extra="ignore")


class QDMpySettings(BaseSettings):
    """Main QDMpy settings class."""

    default_paths: DefaultPathsSettings = Field(
        default_factory=DefaultPathsSettings, description="Default paths"
    )
    odmr: OdmrSettings = Field(default_factory=OdmrSettings, description="ODMR settings")
    model: ModelSettings = Field(default_factory=ModelSettings, description="Model settings")
    fit: FitSettings = Field(default_factory=FitSettings, description="Fitting settings")
    outlier_detection: OutlierDetectionSettings = Field(
        default_factory=OutlierDetectionSettings,
        description="Outlier detection settings",
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings, description="Logging settings"
    )
    nv: NvSettings = Field(default_factory=NvSettings, description="NV centre geometry settings")

    model_config = SettingsConfigDict(
        env_prefix="QDMPY_",
        env_nested_delimiter="__",
        extra="ignore",
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
    """Return True if the pygpufit GPU fitting library can be imported."""
    try:
        import pygpufit.gpufit  # noqa: F401
    except ImportError:
        return False
    else:
        return True
