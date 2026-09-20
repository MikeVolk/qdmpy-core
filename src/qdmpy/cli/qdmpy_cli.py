"""Main CLI implementation for QDMpy.

This module defines the command-line interface structure for QDMpy,
including the argument parsing, subcommands, and execution logic.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from qdmpy.fitting.models import ModelRegistry


def create_parser(version: str) -> argparse.ArgumentParser:
    """Create the argument parser for the QDMpy CLI.

    Args:
        version: The QDMpy version string

    Returns:
        The configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="qdmpy",
        description=f"QDMpy v{version} - Quantum Diamond Microscopy analysis tool",
        epilog="For more information, visit https://mikevolk.github.io/QDMpy",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"QDMpy v{version}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="valid commands",
        help="additional help",
        dest="command",
    )

    models_parser = subparsers.add_parser(
        "models",
        help="List available ODMR fitting models",
        description="Display information about available ODMR fitting models",
    )
    _configure_models_parser(models_parser)

    return parser


def _configure_models_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the parser for the 'models' command.

    Args:
        parser: The argument parser to configure
    """
    parser.add_argument(
        "--detailed",
        help="Show detailed information about models",
        action="store_true",
    )
    parser.add_argument(
        "model_name",
        help="Name of the model to show details for (if not specified, list all models)",
        type=str,
        nargs="?",
    )

    parser.set_defaults(func=models_command_handler)


def process_command(args: argparse.Namespace) -> int:
    """Process a CLI command based on parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if hasattr(args, "func"):
        return args.func(args)

    return 0


def models_command_handler(args: argparse.Namespace) -> int:
    """Handle the 'models' command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    models = ModelRegistry.all()

    if args.model_name:
        if args.model_name not in models:
            logger.error("Model '{}' not found", args.model_name)
            logger.info("Available models: {}", ", ".join(models.keys()))
            return 1

        model_instance = ModelRegistry.get(args.model_name)
        sys.stdout.write(f"Model: {model_instance.name}\n")
        sys.stdout.write(f"N peaks: {model_instance.n_peaks}\n")
        sys.stdout.write(f"Parameters: {model_instance.parameter_names}\n")
        if args.detailed:
            for param in model_instance.parameter_names:
                sys.stdout.write(f"  {param}: {model_instance.units[param]}\n")
    else:
        for name in models:
            model_instance = ModelRegistry.get(name)
            sys.stdout.write(
                f"{name}: {model_instance.n_peaks} peaks, {model_instance.n_parameters} params\n"
            )

    return 0
