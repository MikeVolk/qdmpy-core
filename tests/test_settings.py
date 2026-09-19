"""Tests for QDMpy settings module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from qdmpy.settings import (
    FitSettings,
    LoggingSettings,
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
    get_settings,
    reset_settings,
)


class TestModelConstraintsSettings:
    """Tests for ModelConstraintsSettings."""

    def test_default_constraints(self) -> None:
        """Test default constraint values."""
        settings = ModelConstraintsSettings()
        assert settings.constraint_units == "mt"
        assert settings.center_max_mt == 1.1
        assert settings.center_min_mt == 0.0
        assert settings.width_max_mt == 0.08
        assert settings.width_min_mt == 0.017
        assert settings.center_type == "LOWER_UPPER"
        assert settings.width_type == "LOWER_UPPER"
        assert settings.contrast_min == 0.003
        assert settings.contrast_max == 0
        assert settings.contrast_type == "LOWER"
        assert settings.offset_min == 0
        assert settings.offset_max == 0
        assert settings.offset_type == "FREE"

    def test_custom_constraints_absolute_ghz(self) -> None:
        """Test custom constraint values in absolute GHz mode."""
        settings = ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=1.0,
            center_max=4.0,
            center_type="FREE",
        )
        assert settings.constraint_units == "absolute_ghz"
        assert settings.center_min == 1.0
        assert settings.center_max == 4.0
        assert settings.center_type == "FREE"

    def test_custom_constraints_mt(self) -> None:
        """Test custom mT constraint values."""
        settings = ModelConstraintsSettings(
            constraint_units="mt",
            center_max_mt=10.0,
            width_max_mt=1.5,
        )
        assert settings.center_max_mt == 10.0
        assert settings.width_max_mt == 1.5

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
        assert isinstance(settings.constraints, ModelConstraintsSettings)

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

    def test_structured_logging_disabled_by_default(self) -> None:
        """Structured JSON logging is opt-in.

        Importing a library must not start writing files to the user's home
        directory; only an explicit configure_logging() call installs sinks.
        """
        settings = LoggingSettings()
        assert settings.enable_structured_logging is False

    def test_structured_logging_can_be_enabled(self) -> None:
        """Test that structured logging can be turned on."""
        settings = LoggingSettings(enable_structured_logging=True)
        assert settings.enable_structured_logging is True

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
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.fit, FitSettings)
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

    @staticmethod
    def _write_config(tmp_path: Path, content: str) -> Path:
        config_path = tmp_path / "settings.toml"
        config_path.write_text(content)
        return config_path

    def test_toml_file_loading(self, tmp_path: Path) -> None:
        """Values in settings.toml are actually loaded."""
        config_path = self._write_config(
            tmp_path, '[fit]\nestimator = "LSE"\nmax_number_iterations = 100\n'
        )
        with patch("qdmpy.settings.CONFIG_FILE", config_path):
            settings = QDMpySettings()
        assert settings.fit.estimator == "LSE"
        assert settings.fit.max_number_iterations == 100

    def test_toml_unknown_key_rejected(self, tmp_path: Path) -> None:
        """A typo in settings.toml raises at load time instead of being dropped."""
        config_path = self._write_config(tmp_path, "[fit]\nestimater = 'LSE'\n")
        with (
            patch("qdmpy.settings.CONFIG_FILE", config_path),
            pytest.raises(ValidationError),
        ):
            QDMpySettings()

    def test_env_overrides_toml(self, tmp_path: Path) -> None:
        """Environment variables take priority over settings.toml."""
        config_path = self._write_config(tmp_path, "[fit]\nmax_number_iterations = 100\n")
        with (
            patch("qdmpy.settings.CONFIG_FILE", config_path),
            patch.dict("os.environ", {"QDMPY_FIT__MAX_NUMBER_ITERATIONS": "300"}),
        ):
            settings = QDMpySettings()
        assert settings.fit.max_number_iterations == 300

    def test_init_settings_priority(self) -> None:
        """Test that init settings have highest priority."""
        settings = QDMpySettings(
            fit=FitSettings(estimator="LSE"),
            logging=LoggingSettings(log_level="DEBUG"),
        )
        assert settings.fit.estimator == "LSE"
        assert settings.logging.log_level == "DEBUG"

    def test_unknown_top_level_field_rejected(self) -> None:
        """An unknown top-level key is a typo, not something to swallow."""
        with pytest.raises(ValidationError):
            QDMpySettings(extra_field="should_not_be_ignored")

    def test_unknown_nested_field_rejected(self) -> None:
        """A typo in a nested section must raise, not silently use the default.

        Regression: `center_min_ml` (a typo for `center_min_mt`) used to be
        dropped in silence, so the fit ran under a constraint the user never
        chose and had no way to notice.
        """
        with pytest.raises(ValidationError):
            QDMpySettings(model={"constraints": {"center_min_ml": 0.5}})

    def test_unknown_prefixed_env_var_is_not_rejected(self, monkeypatch) -> None:
        """A stray QDMPY_* env var must not break settings construction.

        pydantic-settings only maps env vars onto declared fields, so
        `extra='forbid'` catches config typos without making unrelated
        environment variables fatal.
        """
        monkeypatch.setenv("QDMPY_TOTALLY_UNRELATED", "x")
        assert isinstance(QDMpySettings(), QDMpySettings)


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
        from qdmpy import get_settings as pkg_get_settings

        assert pkg_get_settings is get_settings


class TestLoggingIsolation:
    """get_settings() must not touch the host application's logging."""

    def test_get_settings_does_not_remove_host_sinks(self) -> None:
        """Reading settings must leave an embedding app's loguru sinks alone.

        Regression: ``get_settings()`` called ``_configure_logging()``, which
        calls ``logger.remove()`` -- dropping every handler in the process.
        Since ``FitManager.__init__`` calls ``get_settings()``, the first fit in
        a GUI/server session silently re-routed that application's logging.
        """
        from loguru import logger

        reset_settings()
        received: list[str] = []
        sink_id = logger.add(received.append, level="INFO")
        try:
            get_settings()
            logger.info("host sink must still be attached")
        finally:
            logger.remove(sink_id)

        assert any("host sink must still be attached" in line for line in received)

    def test_get_settings_creates_no_directories(self, monkeypatch, tmp_path) -> None:
        """Reading settings must not create the config directory as a side effect."""
        import qdmpy.settings as settings_module

        config_path = tmp_path / "config" / "QDMpy"
        monkeypatch.setattr(settings_module, "CONFIG_PATH", config_path)
        monkeypatch.setattr(settings_module, "CONFIG_FILE", config_path / "settings.toml")

        reset_settings()
        get_settings()

        assert not config_path.exists()
