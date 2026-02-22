"""Framework for handling and processing ODMR data.

The `QDMpy.odmr` module provides a comprehensive framework for handling, processing,
and analyzing ODMR (Optically Detected Magnetic Resonance) data. This module is
designed to support modular and extensible workflows, allowing users to efficiently
manage raw data, apply various processing steps, and work with processed data.

Submodules:
    - `data`: Defines the `ODMRData` class, which encapsulates raw and processed
      data, along with metadata and related attributes.
    - `io`: Provides data loaders, such as `MatlabLoader`, for reading ODMR data
      from external sources.
    - `processors`: Includes various processing tools, such as normalization,
      binning, and outlier masking, alongside the `ODMRProcessorManager` for managing
      and applying processing pipelines.
    - `manager`: Contains the `ODMR` class, which acts as the central orchestrator for
      managing raw and processed data, as well as integrating processing workflows.

Key Features:
    - **Data Management**: Encapsulate raw and processed ODMR data in a unified
      structure with metadata.
    - **Flexible Input**: Load data from various sources, including MATLAB files.
    - **Processing Pipelines**: Apply multiple processing steps sequentially,
      including normalization, binning, and outlier masking.
    - **Modular Design**: Extend functionality by implementing custom processors
      or loaders.

Usage Example:
    ```python
    from qdmpy_core.odmr.data import ODMRData
    from qdmpy_core.odmr.io import MatlabLoader
    from qdmpy_core.odmr.processors import NormalizationProcessor, ODMRProcessorManager

    # Load data
    loader = MatlabLoader(data_folder="path/to/matlab/files")
    raw_data, scan_dims, freqs = loader.load()
    odmr_data = ODMRData(data=raw_data, scan_dimensions=scan_dims, frequencies=freqs)

    # Process data
    processor_manager = ODMRProcessorManager()
    processor_manager.add_processor(NormalizationProcessor(method="max"))
    processed_data = processor_manager.process(odmr_data)
    ```

This module is designed for researchers and practitioners working with ODMR data,
offering a robust and extensible solution for their data management and analysis needs.
"""

from __future__ import annotations

from qdmpy_core.odmr.analysis import b111_from_dip_positions
from qdmpy_core.odmr.data import ODMRData
from qdmpy_core.odmr.manager import ODMR
