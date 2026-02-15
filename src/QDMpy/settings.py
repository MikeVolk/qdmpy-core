"""Pydantic settings for QDMpy configuration.

This module defines the complete configuration schema for QDMpy using Pydantic,
supporting TOML files, environment variables, and programmatic overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    InitSettingsSource,
    SecretsSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DefaultPathsSettings(BaseModel):
    """Settings for default paths."""

    data_path: str = Field(default="", description="Default data path")


class OdmrSettings(BaseModel):
    """Settings for ODMR processing."""

    norm_method: Literal["max", "min", "mean"] = Field(
        default="max", description="Normalization method for ODMR data"
    )


class ModelFindPeaksSettings(BaseModel):
    """Settings for model peak finding."""

    prominence: float = Field(default=0.0004, description="Prominence threshold")


class ModelConstraintsSettings(BaseModel):
    """Settings for model fitting constraints."""

    center_min: float = Field(default=2, description="Center frequency minimum")
    center_max: float = Field(default=3.1, description="Center frequency maximum")
    center_type: Literal["FREE", "LOWER", "UPPER", "LOWER_UPPER"] = Field(
        default="LOWER_UPPER", description="Center constraint type"
    )
    width_min: float = Field(default=0.0001, description="Width minimum")
    width_max: float = Field(default=0.005, description="Width maximum")
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
        default="WARNING", description="Log level"
    )

    model_config = ConfigDict(extra="ignore")


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

    model_config = SettingsConfigDict(
        env_prefix="QDMPY_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        _dotenv_settings: DotEnvSettingsSource,
        _file_secret_settings: SecretsSettingsSource,
    ) -> tuple[InitSettingsSource | EnvSettingsSource | TomlConfigSettingsSource | None, ...]:
        """Customize settings sources with TOML file support.

        Priority (highest to lowest):
        1. init_settings (constructor kwargs)
        2. env_settings (environment variables)
        3. TOML file (~/.config/QDMpy/settings.toml)
        4. Default values
        """
        config_file = Path.home() / ".config" / "QDMpy" / "settings.toml"
        toml_settings = (
            TomlConfigSettingsSource(settings_cls, config_file) if config_file.exists() else None
        )

        return tuple(s for s in [init_settings, env_settings, toml_settings] if s is not None)
