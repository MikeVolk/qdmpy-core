# Command Line Interface

qdmpy_core includes a command-line interface (CLI) for performing common tasks without writing Python code.

## Installation

When you install qdmpy_core, the CLI is automatically installed:

```bash
pip install qdmpy_core
```

## Basic Commands

### Processing ODMR Data

Process ODMR data from MATLAB files:

```bash
qdmpy process data.mat --output results.mat
```

Options:
- `--bin-factor`: Spatial binning factor (default: 1)
- `--normalize`: Normalize the data (choices: max, mean, minmax)
- `--remove-outliers`: Remove outlier pixels (threshold value)
- `--output`: Output file path (default: processed_data.mat)

### Calculating QDM Images

Calculate magnetic field images from ODMR data:

```bash
qdmpy calculate data.mat --output field_map.mat
```

Options:
- `--model`: Spectral model to use (choices: ESRSINGLE, ESR14N, ESR15N)
- `--bin-factor`: Spatial binning factor (default: 1)
- `--normalize`: Normalize the data (choices: max, mean, minmax)
- `--output`: Output file path (default: field_map.mat)

## Advanced Usage

### Configuration File

You can specify a configuration file to control the processing pipeline:

```bash
qdmpy process data.mat --config my_config.ini
```

Example configuration file:
```ini
[processing]
bin_factor = 2
normalize = max
remove_outliers = 3.0

[fitting]
model = ESR14N
center_min = 2.87e9
center_max = 2.88e9
width_min = 1e6
contrast_max = 0.3
```

### Batch Processing

Process multiple files at once:

```bash
qdmpy process data1.mat data2.mat data3.mat --output results/
```

This will process each file and save the results in the specified directory.

## Examples

### Basic Processing with Default Settings

```bash
qdmpy process experiment/run_00001.mat
```

### Full Analysis Pipeline

```bash
# Process the data
qdmpy process experiment/run_00001.mat --bin-factor 2 --normalize max --output processed.mat

# Calculate field maps
qdmpy calculate processed.mat --model ESR14N --output field_map.mat
```

For more details on the available commands and options, use the `--help` flag:

```bash
qdmpy --help
qdmpy process --help
qdmpy calculate --help
```