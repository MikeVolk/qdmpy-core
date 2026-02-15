#!/usr/bin/python
"""Command-line interface for QDMpy data processing.

This module provides the original command-line interface for processing
Optically Detected Magnetic Resonance (ODMR) data from Quantum Diamond Microscopy
(QDM) measurements.

Note: This interface is maintained for backwards compatibility.
New users should use the 'qdmpy' command-line interface instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import QDMpy
from loguru import logger
from QDMpy._core.qdm_old import QDM


def main(argv: list[str] = None) -> int:
    """Main function for the legacy QDMpy command line interface.

    Processes command line arguments to calculate B111 field from ODMR data
    recorded with QDMio made QDM.

    Args:
        argv: List of command line arguments. If None, uses sys.argv[1:].

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    if argv is None:
        argv = sys.argv[1:]

    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Calculate the B111 field from ODMR data recorded with QDMio made QDM",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input path, location of the QDM data files and LED/laser images.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for results (default: input_path/results)",
        required=False,
    )
    parser.add_argument(
        "-b",
        "--binfactor",
        type=int,
        help="Binning factor of the ODMR data. Default: 1",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Type of model used in the experiment. Default: 'auto'",
        default="auto",
        required=False,
    )
    parser.add_argument(
        "-gf",
        "--globalfluorescence",
        type=float,
        help="Global fluorescence of the sample. Default: 0.2",
        default=0.2,
        required=False,
    )
    parser.add_argument(
        "--debug",
        help="Sets logging to DEBUG level",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing results",
        action="store_true",
        default=False,
        required=False,
    )

    try:
        args = parser.parse_args(argv)

        # Show warning about deprecated interface
        logger.warning(
            "This command-line interface is deprecated. " "Please use 'qdmpy process' instead."
        )

        # Determine output directory
        output_dir = args.output if args.output else Path(args.input) / "results"
        output_path = Path(output_dir)

        # Check if output directory exists
        if output_path.exists() and not args.overwrite:
            logger.error(
                f"Output directory {output_dir} already exists. Use --overwrite to overwrite."
            )
            return 1

        # Create output directory if it doesn't exist
        if not output_path.exists():
            logger.info(f"Creating output directory: {output_dir}")
            output_path.mkdir(parents=True)

        # Log the command parameters
        logger.info(f"Processing data from: {args.input}")
        logger.info(f"Binning factor: {args.binfactor}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Global fluorescence: {args.globalfluorescence}")

        # Create QDM object from data
        qdm_obj = QDM.from_qdmio(args.input, model_name=args.model)

        # Apply binning if requested
        if args.binfactor > 1:
            logger.info(f"Applying spatial binning with factor {args.binfactor}...")
            qdm_obj.bin_data(bin_factor=args.binfactor)

        # Apply global fluorescence correction
        logger.info(f"Applying global fluorescence correction ({args.globalfluorescence})...")
        qdm_obj.correct_glob_fluorescence(glob_fluorescence=args.globalfluorescence)

        # Fit ODMR data
        logger.info("Fitting ODMR spectra...")
        qdm_obj.fit_odmr()

        # Export results
        logger.info(f"Exporting results to {output_dir}...")
        qdm_obj.export_qdmio(output_path=output_dir)

        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed successfully in {elapsed_time:.2f} seconds")
        return 0

    except Exception as e:
        logger.error(f"Error processing data: {e!s}")
        if getattr(args, "debug", False):
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
