# ODMR Processors

The ODMR processors module provides signal processing algorithms for ODMR data.

## BaseProcessor

The `BaseProcessor` is an abstract base class for all ODMR processors.

::: qdmpy_core.odmr.processors.BaseProcessor
    options:
      show_root_heading: true
      show_source: true

## NormalizationProcessor

The `NormalizationProcessor` normalizes ODMR data to a common scale.

::: qdmpy_core.odmr.processors.NormalizationProcessor
    options:
      show_root_heading: true
      show_source: true

## BinningProcessor

The `BinningProcessor` performs spatial binning on ODMR data.

::: qdmpy_core.odmr.processors.BinningProcessor
    options:
      show_root_heading: true
      show_source: true

## OutlierProcessor

The `OutlierProcessor` identifies and removes outlier pixels.

::: qdmpy_core.odmr.processors.OutlierProcessor
    options:
      show_root_heading: true
      show_source: true

## FluorescenceCorrectionProcessor

The `FluorescenceCorrectionProcessor` corrects for global fluorescence variations.

::: qdmpy_core.odmr.processors.FluorescenceCorrectionProcessor
    options:
      show_root_heading: true
      show_source: true

## ODMRProcessorManager

The `ODMRProcessorManager` coordinates multiple processors in a pipeline.

::: qdmpy_core.odmr.processors.ODMRProcessorManager
    options:
      show_root_heading: true
      show_source: true
