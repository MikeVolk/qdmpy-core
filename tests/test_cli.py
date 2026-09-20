"""Tests for the CLI package (cli/__init__.py and cli/qdmpy_cli.py)."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from qdmpy.cli import main
from qdmpy.cli.qdmpy_cli import (
    create_parser,
    models_command_handler,
    process_command,
)


class TestCreateParser:
    """Test argument parser creation."""

    def test_creates_parser(self) -> None:
        parser = create_parser("1.0.0")
        assert isinstance(parser, argparse.ArgumentParser)

    def test_version_in_description(self) -> None:
        parser = create_parser("2.3.4")
        assert "2.3.4" in parser.description

    def test_models_subcommand_exists(self) -> None:
        parser = create_parser("1.0.0")
        args = parser.parse_args(["models"])
        assert args.command == "models"
        assert hasattr(args, "func")

    def test_models_with_name(self) -> None:
        parser = create_parser("1.0.0")
        args = parser.parse_args(["models", "ESR14N"])
        assert args.model_name == "ESR14N"

    def test_models_detailed_flag(self) -> None:
        parser = create_parser("1.0.0")
        args = parser.parse_args(["models", "--detailed"])
        assert args.detailed is True

    def test_debug_flag(self) -> None:
        parser = create_parser("1.0.0")
        args = parser.parse_args(["--debug", "models"])
        assert args.debug is True


class TestProcessCommand:
    """Test command dispatch."""

    def test_calls_func(self) -> None:
        args = argparse.Namespace(func=lambda _a: 0)
        assert process_command(args) == 0

    def test_returns_zero_without_func(self) -> None:
        args = argparse.Namespace()
        assert process_command(args) == 0


class TestModelsCommandHandler:
    """Test the models subcommand."""

    def test_list_all_models(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(model_name=None, detailed=False)
        rc = models_command_handler(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "ESR14N" in out

    def test_show_specific_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(model_name="ESR14N", detailed=False)
        rc = models_command_handler(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "ESR14N" in out
        assert "Parameters" in out

    def test_show_detailed(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(model_name="ESR14N", detailed=True)
        rc = models_command_handler(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "center" in out

    def test_unknown_model(self) -> None:
        args = argparse.Namespace(model_name="NONEXISTENT", detailed=False)
        rc = models_command_handler(args)
        assert rc == 1


class TestMain:
    """Test the main() entry point."""

    def test_no_args_returns_2(self) -> None:
        with patch("sys.argv", ["qdmpy"]):
            rc = main()
        assert rc == 2

    def test_models_command(self) -> None:
        with patch("sys.argv", ["qdmpy", "models"]):
            rc = main()
        assert rc == 0

    def test_keyboard_interrupt(self) -> None:
        with (
            patch("sys.argv", ["qdmpy", "models"]),
            patch(
                "qdmpy.cli.qdmpy_cli.process_command",
                side_effect=KeyboardInterrupt,
            ),
        ):
            rc = main()
        assert rc == 130

    def test_generic_error(self) -> None:
        with (
            patch("sys.argv", ["qdmpy", "models"]),
            patch(
                "qdmpy.cli.qdmpy_cli.process_command",
                side_effect=RuntimeError("boom"),
            ),
        ):
            rc = main()
        assert rc == 1

    def test_version_fallback(self) -> None:
        with (
            patch("sys.argv", ["qdmpy", "models"]),
            patch("qdmpy.cli.get_version", side_effect=PackageNotFoundError("no pkg")),
        ):
            rc = main()
        assert rc == 0

    def test_resolves_real_version_not_unknown(self) -> None:
        """Regression test: get_version() used to be called with "QDMpy",
        but the distribution is named "qdmpy-core" (pyproject.toml), so a
        real install always raised PackageNotFoundError and every --version
        / error-path banner silently showed "unknown".
        """
        from importlib.metadata import version

        assert version("qdmpy-core") != "unknown"

        with patch("sys.argv", ["qdmpy", "models"]), patch("qdmpy.cli.get_version") as mock_gv:
            mock_gv.return_value = "9.9.9"
            main()

        mock_gv.assert_called_once_with("qdmpy-core")
