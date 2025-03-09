#!/usr/bin/python
"""Command-line interface for QDMpy data processing.

This module provides a comprehensive command-line interface for processing
Optically Detected Magnetic Resonance (ODMR) data from Quantum Diamond Microscopy
(QDM) measurements. Key features include:

- Batch processing: Handling multiple data files in a single command
- Parameter specification: Setting processing parameters via command-line arguments
- Model selection: Choosing appropriate fitting models for ODMR spectra
- Output control: Configuring output formats and locations
- Processing customization: Setting binning factors, fluorescence correction, etc.
- Help system: Detailed documentation accessible through --help flags

This interface makes QDMpy functionality accessible without requiring Python
programming, allowing users to integrate QDM data processing into scripts
and workflows.
"""
from __future__ import annotations

import argparse
import sys
import time

from argdoc import generate_doc

import QDMpy
from src.QDMpy._core.qdm_old import QDM


@generate_doc
def main(argv: list[str]) -> None:
    """Main function for the QDMpy command line interface.

    Processes command line arguments to calculate B111 field from ODMR data
    recorded with QDMio made QDM.

    Args:
        argv: List of command line arguments.

    Returns:
        None
    """
    tstart = time.process_time()

    parser = argparse.ArgumentParser(
        description='Calculate the B111 field from ODMR data recorded with QDMio made QDM',
    )
    parser.add_argument(
        '-i',
        '--input',
        help='input path, location of the QDM data files and LED/laser images.',
        required=True,
    )
    parser.add_argument(
        '-b',
        '--binfactor',
        type=int,
        help='Binning factor of the ODMR data. Default: 1',
        default=1,
        required=False,
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help="Type of model used in the experiment. Default: 'auto'",
        default='auto',
        required=False,
    )
    parser.add_argument(
        '-gf',
        '--globalfluorescence',
        type=float,
        help='Global fluorescence of the sample. Default: 0.2',
        default=0.2,
        required=False,
    )
    parser.add_argument(
        '--debug',
        help='sets logging to DEBUG level',
        action='store_true',
        default=False,
        required=False,
    )

    args = parser.parse_args()

    if args.debug:
        QDMpy.LOG.setLevel('DEBUG')
    else:
        QDMpy.LOG.setLevel('INFO')

    qdm_obj = QDM.from_qdmio(args.input, model_name=args.model)
    qdm_obj.bin_data(bin_factor=args.binfactor)
    qdm_obj.correct_glob_fluorescence(glob_fluorescence=args.globalfluorescence)
    qdm_obj.fit_odmr()
    qdm_obj.export_qdmio()
    QDMpy.LOG.info(f'QDMpy finished in {time.process_time() - tstart:.2f} seconds')


if __name__ == '__main__':
    main(sys.argv[1:])
