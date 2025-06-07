"""Main CLI implementation for QDMpy.

This module defines the command-line interface structure for QDMpy,
including the argument parsing, subcommands, and execution logic.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional


import QDMpy
from QDMpy.cli import CLI_LOGGER
from QDMpy.models import ModelRegistry


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the QDMpy CLI.
    
    Args:
        debug: Whether to enable debug logging
    """
    # Configure the QDMpy root logger
    level = logging.DEBUG if debug else logging.INFO
    QDMpy.LOG.setLevel(level)
    CLI_LOGGER.setLevel(level)
    
    if debug:
        CLI_LOGGER.debug("Debug logging enabled")


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
        "-o", "--output",
        help="Output directory for results (default: input_path/results)",
        type=str,
    )
    parser.add_argument(
        "-b", "--bin-factor",
        help="Spatial binning factor (default: 1)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-m", "--model",
        help="ODMR fitting model (default: auto)",
        type=str,
        choices=list(ModelRegistry.all().keys()) + ["auto"],
        default="auto",
    )
    parser.add_argument(
        "-gf", "--global-fluorescence",
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
    # Set up logging based on debug flag
    setup_logging(args.debug)
    
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
    start_time = time.time()
    
    # Log the command parameters
    CLI_LOGGER.info(f"Processing data from: {args.input_path}")
    CLI_LOGGER.info(f"Binning factor: {args.bin_factor}")
    CLI_LOGGER.info(f"Model: {args.model}")
    CLI_LOGGER.info(f"Global fluorescence: {args.global_fluorescence}")
    
    # Determine output directory
    output_dir = args.output if args.output else os.path.join(args.input_path, "results")
    output_path = Path(output_dir)
    
    # Check if output directory exists
    if output_path.exists() and not args.overwrite:
        CLI_LOGGER.error(f"Output directory {output_dir} already exists. Use --overwrite to overwrite.")
        return 1
    
    # Create output directory if it doesn't exist
    if not output_path.exists():
        CLI_LOGGER.info(f"Creating output directory: {output_dir}")
        output_path.mkdir(parents=True)
    
    try:
        # Import here to avoid slow imports when running other commands
        from QDMpy._core.qdm_old import QDM
        
        CLI_LOGGER.info("Loading and processing data...")
        
        # Create QDM object from data
        qdm_obj = QDM.from_qdmio(args.input_path, model_name=args.model)
        
        # Apply binning if requested
        if args.bin_factor > 1:
            CLI_LOGGER.info(f"Applying spatial binning with factor {args.bin_factor}...")
            qdm_obj.bin_data(bin_factor=args.bin_factor)
        
        # Apply global fluorescence correction
        CLI_LOGGER.info(f"Applying global fluorescence correction ({args.global_fluorescence})...")
        qdm_obj.correct_glob_fluorescence(glob_fluorescence=args.global_fluorescence)
        
        # Fit ODMR data
        CLI_LOGGER.info("Fitting ODMR spectra...")
        qdm_obj.fit_odmr()
        
        # Export results
        CLI_LOGGER.info(f"Exporting results to {output_dir}...")
        qdm_obj.export_qdmio(output_path=output_dir)
        
        # Generate plots if not disabled
        if not args.no_plots:
            CLI_LOGGER.info("Generating plots...")
            # Add plot generation here
        
        elapsed_time = time.time() - start_time
        CLI_LOGGER.info(f"Processing completed successfully in {elapsed_time:.2f} seconds")
        return 0
        
    except Exception as e:
        CLI_LOGGER.error(f"Error processing data: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


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
            CLI_LOGGER.error(f"Model '{args.model_name}' not found")
            CLI_LOGGER.info(f"Available models: {', '.join(models.keys())}")
            return 1
        
        model_info = models[args.model_name]
        model_instance = ModelRegistry.get(args.model_name)
        
        print(f"\nModel: {args.model_name}")
        print(f"Hyperfine constant: {model_info.get('hyp', 'N/A')} GHz")
        print(f"Number of peaks: {model_instance.n_peaks}")
        print(f"Number of parameters: {model_instance.n_parameters}")
        print(f"Parameters: {', '.join(model_instance.parameters_unique)}")
        
        if args.detailed:
            # Add more detailed information here
            print("\nDetailed Parameter Information:")
            for param in model_instance.parameters_unique:
                print(f"  - {param}")
    else:
        # List all available models
        print("\nAvailable ODMR Models:")
        for name, info in models.items():
            model = ModelRegistry.get(name)
            print(f"  - {name}: {model.n_peaks} peaks, {model.n_parameters} parameters")
        
        print("\nUse 'qdmpy models <model_name>' to see details for a specific model")
    
    return 0


def info_command_handler(args: argparse.Namespace) -> int:
    """Handle the 'info' command.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    path = Path(args.data_path)
    
    if not path.exists():
        CLI_LOGGER.error(f"Path does not exist: {path}")
        return 1
    
    try:
        # Import the loader
        from QDMpy.odmr.io import MatlabLoader
        
        if path.is_dir():
            CLI_LOGGER.info(f"Analyzing directory: {path}")
            loader = MatlabLoader(data_folder=str(path))
            file_info = loader.get_file_list()
            
            print(f"\nDirectory: {path}")
            print(f"Found {len(file_info)} data files")
            
            if not args.summary:
                for i, file in enumerate(file_info):
                    print(f"\nFile {i+1}: {file}")
        else:
            CLI_LOGGER.info(f"Analyzing file: {path}")
            print(f"\nFile: {path}")
            print(f"Size: {path.stat().st_size / (1024*1024):.2f} MB")
            print(f"Last modified: {time.ctime(path.stat().st_mtime)}")
            
            # Add more file-specific analysis here
        
        return 0
        
    except Exception as e:
        CLI_LOGGER.error(f"Error analyzing data: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
