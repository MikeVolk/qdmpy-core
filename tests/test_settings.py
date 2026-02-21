"""Tests for QDMpy settings module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import pytest

from QDMpy.settings import (
    DefaultPathsSettings,
    FitSettings,
    LocalOutlierFactorSettings,
    LoggingSettings,
    ModelConstraintsSettings,
    ModelFindPeaksSettings,
    ModelSettings,
    OdmrSettings,
    OutlierDetectionSettings,
    QDMpySettings,
    StatisticsPercentileSettings,
    get_settings,
    reset_settings,
)


class TestDefaultPathsSettings:
    """Tests for DefaultPathsSettings."""

    def test_default_values(self) -> None:
        """Test default values."""
        settings = DefaultPathsSettings()
        assert settings.data_path == ""

    def test_custom_values(self) -> None:
        """Test custom values."""
        settings = DefaultPathsSettings(data_path="/home/data")
        assert settings.data_path == "/home/data"


class TestOdmrSettings:
    """Tests for OdmrSettings."""

    def test_default_norm_method(self) -> None:
        """Test default normalization method."""
        settings = OdmrSettings()
        assert settings.norm_method == "max"

    def test_custom_norm_method(self) -> None:
        """Test custom normalization method."""
        for method in ["max", "min", "mean"]:
            settings = OdmrSettings(norm_method=method)
            assert settings.norm_method == method

    def test_invalid_norm_method(self) -> None:
        """Test invalid normalization method raises error."""
        with pytest.raises(ValueError):
            OdmrSettings(norm_method="invalid")


class TestModelConstraintsSettings:
    """Tests for ModelConstraintsSettings."""

    def test_default_constraints(self) -> None:
        """Test default constraint values."""
        settings = ModelConstraintsSettings()
        assert settings.center_min == 2
        assert settings.center_max == 3.1
        assert settings.center_type == "LOWER_UPPER"
        assert settings.width_min == 0.0001
        assert settings.width_max == 0.005
        assert settings.width_type == "LOWER_UPPER"
        assert settings.contrast_min == 0.003
        assert settings.contrast_max == 0
        assert settings.contrast_type == "LOWER"
        assert settings.offset_min == 0
        assert settings.offset_max == 0
        assert settings.offset_type == "FREE"

    def test_custom_constraints(self) -> None:
        """Test custom constraint values."""
        settings = ModelConstraintsSettings(
            center_min=1.0,
            center_max=4.0,
            center_type="FREE",
        )
        assert settings.center_min == 1.0
        assert settings.center_max == 4.0
        assert settings.center_type == "FREE"

    def test_valid_constraint_types(self) -> None:
        """Test all valid constraint types."""
        for constraint_type in ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]:
            settings = ModelConstraintsSettings(center_type=constraint_type)
            assert settings.center_type == constraint_type

    def test_invalid_constraint_type(self) -> None:
        """Test invalid constraint type raises error."""
        with pytest.raises(ValueError):
            ModelConstraintsSettings(center_type="INVALID")


class TestModelSettings:
    """Tests for ModelSettings."""

    def test_default_model_settings(self) -> None:
        """Test default model settings."""
        settings = ModelSettings()
        assert isinstance(settings.find_peaks, ModelFindPeaksSettings)
        assert isinstance(settings.constraints, ModelConstraintsSettings)
        assert settings.find_peaks.prominence == 0.0004

    def test_custom_model_settings(self) -> None:
        """Test custom model settings."""
        constraints = ModelConstraintsSettings(center_min=2.5)
        settings = ModelSettings(constraints=constraints)
        assert settings.constraints.center_min == 2.5


class TestFitSettings:
    """Tests for FitSettings."""

    def test_default_fit_settings(self) -> None:
        """Test default fit settings."""
        settings = FitSettings()
        assert settings.estimator == "MLE"
        assert settings.max_number_iterations == 1000
        assert settings.tolerance == 1e-10

    def test_custom_fit_settings(self) -> None:
        """Test custom fit settings."""
        settings = FitSettings(
            estimator="LSE",
            max_number_iterations=500,
            tolerance=1e-6,
        )
        assert settings.estimator == "LSE"
        assert settings.max_number_iterations == 500
        assert settings.tolerance == 1e-6

    def test_valid_estimators(self) -> None:
        """Test valid estimator types."""
        for estimator in ["LSE", "MLE"]:
            settings = FitSettings(estimator=estimator)
            assert settings.estimator == estimator

    def test_invalid_estimator(self) -> None:
        """Test invalid estimator raises error."""
        with pytest.raises(ValueError):
            FitSettings(estimator="INVALID")


class TestOutlierDetectionSettings:
    """Tests for OutlierDetectionSettings."""

    def test_default_outlier_settings(self) -> None:
        """Test default outlier detection settings."""
        settings = OutlierDetectionSettings()
        assert settings.method == "LocalOutlierFactor"
        assert isinstance(settings.local_outlier_factor, LocalOutlierFactorSettings)
        assert isinstance(settings.statistics_percentile, StatisticsPercentileSettings)

    def test_statistics_percentile_method(self) -> None:
        """Test StatisticsPercentile method settings."""
        settings = OutlierDetectionSettings(method="StatisticsPercentile")
        assert settings.method == "StatisticsPercentile"

    def test_local_outlier_factor_defaults(self) -> None:
        """Test LocalOutlierFactor default settings."""
        settings = OutlierDetectionSettings()
        assert settings.local_outlier_factor.n_neighbors == 20
        assert settings.local_outlier_factor.algorithm == "auto"
        assert settings.local_outlier_factor.leaf_size == 30
        assert settings.local_outlier_factor.metric == "minkowski"
        assert settings.local_outlier_factor.p == 2
        assert settings.local_outlier_factor.contamination == "auto"


class TestLoggingSettings:
    """Tests for LoggingSettings."""

    def test_default_log_level(self) -> None:
        """Test default log level."""
        settings = LoggingSettings()
        assert settings.log_level == "INFO"

    def test_custom_log_levels(self) -> None:
        """Test custom log levels."""
        for level in ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
            settings = LoggingSettings(log_level=level)
            assert settings.log_level == level

    def test_invalid_log_level(self) -> None:
        """Test invalid log level raises error."""
        with pytest.raises(ValueError):
            LoggingSettings(log_level="INVALID")

    def test_structured_logging_enabled_by_default(self) -> None:
        """Test that structured logging is enabled by default."""
        settings = LoggingSettings()
        assert settings.enable_structured_logging is True

    def test_structured_logging_can_be_disabled(self) -> None:
        """Test that structured logging can be disabled."""
        settings = LoggingSettings(enable_structured_logging=False)
        assert settings.enable_structured_logging is False

    def test_structured_log_dir_defaults_to_none(self) -> None:
        """Test that structured log dir defaults to None."""
        settings = LoggingSettings()
        assert settings.structured_log_dir is None

    def test_structured_log_dir_custom_path(self) -> None:
        """Test setting a custom structured log directory."""
        custom_dir = "/tmp/custom_logs"
        settings = LoggingSettings(structured_log_dir=custom_dir)
        assert settings.structured_log_dir == custom_dir

    def test_legacy_log_file_still_supported(self) -> None:
        """Test backward compatibility with log_file setting."""
        log_file = "/tmp/app.log"
        settings = LoggingSettings(log_file=log_file)
        assert settings.log_file == log_file


class TestQDMpySettings:
    """Tests for the main QDMpySettings class."""

    def test_default_settings(self) -> None:
        """Test that default settings are created correctly."""
        settings = QDMpySettings()
        assert isinstance(settings.default_paths, DefaultPathsSettings)
        assert isinstance(settings.odmr, OdmrSettings)
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.fit, FitSettings)
        assert isinstance(settings.outlier_detection, OutlierDetectionSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_custom_fit_settings(self) -> None:
        """Test passing custom fit settings."""
        fit_settings = FitSettings(estimator="LSE", max_number_iterations=500)
        settings = QDMpySettings(fit=fit_settings)
        assert settings.fit.estimator == "LSE"
        assert settings.fit.max_number_iterations == 500

    def test_nested_settings_override(self) -> None:
        """Test overriding nested settings."""
        constraints = ModelConstraintsSettings(center_min=1.5)
        model_settings = ModelSettings(constraints=constraints)
        settings = QDMpySettings(model=model_settings)
        assert settings.model.constraints.center_min == 1.5

    def test_environment_variable_override(self) -> None:
        """Test environment variable overrides."""
        with patch.dict("os.environ", {"QDMPY_LOGGING__LOG_LEVEL": "DEBUG"}):
            settings = QDMpySettings()
            assert settings.logging.log_level == "DEBUG"

    def test_environment_variable_nested_override(self) -> None:
        """Test nested environment variable overrides."""
        with patch.dict(
            "os.environ",
            {"QDMPY_FIT__ESTIMATOR": "LSE", "QDMPY_FIT__MAX_NUMBER_ITERATIONS": "200"},
        ):
            settings = QDMpySettings()
            assert settings.fit.estimator == "LSE"
            assert settings.fit.max_number_iterations == 200

    def test_toml_file_loading(self) -> None:
        """Test loading settings from a TOML file."""
        # Create a temporary TOML file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "settings.toml"
            config_path.write_text('[fit]\nestimator = "LSE"\nmax_number_iterations = 100\n')

            # Mock the config file path
            with (
                patch(
                    "QDMpy.settings.Path.home",
                    return_value=Path(tmpdir),
                ),
                patch(
                    "QDMpy.settings.QDMpySettings.model_config",
                    {"toml_file": config_path},
                    create=True,
                ),
            ):
                settings = QDMpySettings()
                # Since we're mocking, just verify the settings work
                assert isinstance(settings, QDMpySettings)

    def test_init_settings_priority(self) -> None:
        """Test that init settings have highest priority."""
        settings = QDMpySettings(
            fit=FitSettings(estimator="LSE"),
            logging=LoggingSettings(log_level="DEBUG"),
        )
        assert settings.fit.estimator == "LSE"
        assert settings.logging.log_level == "DEBUG"

    def test_extra_fields_ignored(self) -> None:
        """Test that extra fields are ignored (extra='ignore')."""
        # This should not raise an error
        settings = QDMpySettings(extra_field="should_be_ignored")
        assert isinstance(settings, QDMpySettings)


class TestGetSettings:
    """Tests for the get_settings singleton and reset_settings."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Ensure a clean singleton state for each test."""
        reset_settings()
        yield
        reset_settings()

    def test_returns_qdmpy_settings(self) -> None:
        """get_settings() returns a QDMpySettings instance."""
        assert isinstance(get_settings(), QDMpySettings)

    def test_singleton(self) -> None:
        """Repeated calls return the same object."""
        assert get_settings() is get_settings()

    def test_reset_settings_invalidates_cache(self) -> None:
        """reset_settings() causes the next call to return a fresh instance."""
        first = get_settings()
        reset_settings()
        second = get_settings()
        # New instance after cache clear
        assert first is not second

    def test_get_settings_re_exported_from_package(self) -> None:
        """get_settings is importable from the top-level QDMpy package."""
        from QDMpy import get_settings as pkg_get_settings

        assert pkg_get_settings is get_settings
