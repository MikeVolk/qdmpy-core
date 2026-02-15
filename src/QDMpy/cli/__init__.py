# noqa: N999
"""Command-line interface package for QDMpy.

This package provides command-line tools for interacting with QDMpy functionality.
The entry point 'qdmpy' provides access to various subcommands for processing and
analyzing Quantum Diamond Microscopy data.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as get_version

from loguru import logger


# Define the main entry point function that will be called by the 'qdmpy' command
def main() -> int:
    """Main entry point for the QDMpy CLI.

    This function delegates to subcommands defined in the CLI package.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    from .qdmpy_cli import create_parser, process_command

    # Get QDMpy version
    try:
        qdmpy_version = get_version("QDMpy")
    except Exception:
        qdmpy_version = "unknown"

    # Create the argument parser
    parser = create_parser(qdmpy_version)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command provided
    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    # Process the command
    try:
        return process_command(args)
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e!s}")
        if args.debug:
            # Print full traceback in debug mode
            import traceback

            traceback.print_exc()
        return 1
