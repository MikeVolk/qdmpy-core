"""Main CLI implementation for QDMpy.

This module defines the command-line interface structure for QDMpy,
including the argument parsing, subcommands, and execution logic.
"""

from __future__ import annotations

import argparse

from loguru import logger

from QDMpy.fitting.models import ModelRegistry


def create_parser(version: str) -> argparse.ArgumentParser:
    """Create the argument parser for the QDMpy CLI.

    Args:
        version: The QDMpy version string

    Returns:
        The configured argument parser
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        prog="qdmpy",
        description=f"QDMpy v{version} - Quantum Diamond Microscopy analysis tool",
        epilog="For more information, visit https://mikevolk.github.io/QDMpy",
    )

    # Add global options
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

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="valid commands",
        help="additional help",
        dest="command",
    )

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process ODMR data and calculate B111 field",
        description="Process ODMR data from QDMio and calculate B111 magnetic field maps",
    )
    _configure_process_parser(process_parser)

    # Models command
    models_parser = subparsers.add_parser(
        "models",
        help="List available ODMR fitting models",
        description="Display information about available ODMR fitting models",
    )
    _configure_models_parser(models_parser)

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about data files",
        description="Display information about QDM data files",
    )
    _configure_info_parser(info_parser)

    return parser


def _configure_process_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the parser for the 'process' command.

    Args:
        parser: The argument parser to configure
    """
    parser.add_argument(
        "input_path",
        help="Path to the input data folder containing QDMio files",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for results (default: input_path/results)",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--bin-factor",
        help="Spatial binning factor (default: 1)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="ODMR fitting model (default: auto)",
        type=str,
        choices=[*list(ModelRegistry.all().keys()), "auto"],
        default="auto",
    )
    parser.add_argument(
        "-gf",
        "--global-fluorescence",
        help="Global fluorescence correction value (default: 0.2)",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing results",
        action="store_true",
    )
    parser.add_argument(
        "--no-plots",
        help="Disable generation of plots",
        action="store_true",
    )

    parser.set_defaults(func=process_command_handler)


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


def _configure_info_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the parser for the 'info' command.

    Args:
        parser: The argument parser to configure
    """
    parser.add_argument(
        "data_path",
        help="Path to data file or directory",
        type=str,
    )
    parser.add_argument(
        "--summary",
        help="Show only summary information",
        action="store_true",
    )

    parser.set_defaults(func=info_command_handler)


def process_command(args: argparse.Namespace) -> int:
    """Process a CLI command based on parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Execute the appropriate command handler
    if hasattr(args, "func"):
        return args.func(args)

    return 0


def process_command_handler(args: argparse.Namespace) -> int:
    """Handle the 'process' command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    raise NotImplementedError(
        "The 'process' command is not yet available in the refactored codebase. "
        "See QDMpy documentation for the current API."
    )


def models_command_handler(args: argparse.Namespace) -> int:
    """Handle the 'models' command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    models = ModelRegistry.all()

    if args.model_name:
        # Show details for a specific model
        if args.model_name not in models:
            logger.error(f"Model '{args.model_name}' not found")
            logger.info(f"Available models: {', '.join(models.keys())}")
            return 1

        models[args.model_name]
        model_instance = ModelRegistry.get(args.model_name)

        if args.detailed:
            # Add more detailed information here
            for _param in model_instance.parameter_names:
                pass
    else:
        # List all available models
        for name, _info in models.items():
            ModelRegistry.get(name)

    return 0


def info_command_handler(args: argparse.Namespace) -> int:
    """Handle the 'info' command.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    raise NotImplementedError(
        "The 'info' command is not yet available in the refactored codebase. "
        "See QDMpy documentation for the current API."
    )
