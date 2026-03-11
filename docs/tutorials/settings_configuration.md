# QDMpy Settings Configuration Guide

This guide explains how QDMpy's Pydantic-based settings system works, including the complete configuration hierarchy, how to customize settings, and best practices for different use cases.

## Table of Contents

1. [Settings Hierarchy](#settings-hierarchy)
2. [Configuration Methods](#configuration-methods)
3. [Default Values](#default-values)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)

---

## Settings Hierarchy

QDMpy uses a **nested Pydantic model** structure to organize all configuration options. The hierarchy reflects the different functional areas of the package:

```
QDMpySettings (root)
├── default_paths
│   └── data_path: str
├── odmr
│   └── norm_method: Literal['mean']
├── model
│   ├── find_peaks
│   │   └── prominence: float
│   └── constraints
│       ├── constraint_units ('mt' | 'absolute_ghz')
│       ├── center_min/max/type (absolute_ghz mode)
│       ├── width_min/max/type (absolute_ghz mode)
│       ├── center_min_mt/center_max_mt (mt mode)
│       ├── width_min_mt/width_max_mt (mt mode)
│       ├── contrast_min/max/type
│       └── offset_min/max/type
├── fit
│   ├── estimator: Literal['LSE', 'MLE']
│   ├── max_number_iterations: int
│   └── tolerance: float
├── outlier_detection
│   ├── method: Literal['LocalOutlierFactor', 'StatisticsPercentile']
│   ├── statistics_percentile
│   │   ├── chi2_percentile: list[float]
│   │   ├── width_percentile: list[float]
│   │   └── contrast_percentile: list[float]
│   └── local_outlier_factor
│       ├── n_neighbors: int
│       ├── algorithm: str
│       ├── leaf_size: int
│       ├── metric: str
│       ├── p: int
│       └── contamination: str | float
└── logging
    └── log_level: Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
```

### Understanding the Structure

Each level of the hierarchy is a **Pydantic model** that validates its contents:

- **Top level** (`QDMpySettings`): Manages the complete configuration
- **Second level** (e.g., `FitSettings`, `ModelSettings`): Groups related settings
- **Leaf values**: Individual configuration parameters with types and validation

This structure provides:
- ✓ Type safety (all values are validated)
- ✓ Clear organization (settings grouped by functionality)
- ✓ Attribute access (e.g., `settings.fit.estimator` instead of `settings['fit']['estimator']`)
- ✓ IDE autocomplete support

---

## Configuration Methods

QDMpy supports **three configuration methods**, applied in this priority order:

### 1. Programmatic Overrides (Highest Priority)

Pass settings directly as constructor arguments:

```python
from qdmpy import get_settings
from qdmpy.settings import (
    QDMpySettings,
    FitSettings,
    ModelSettings,
    ModelConstraintsSettings,
)

# Override specific settings when creating a new instance
custom_settings = QDMpySettings(
    fit=FitSettings(
        estimator='LSE',
        max_number_iterations=500,
        tolerance=1e-8,
    ),
)

print(custom_settings.fit.estimator)  # Output: LSE

SETTINGS = get_settings()
```

### 2. Environment Variables

Set environment variables with the prefix `QDMPY_` and nested delimiter `__`:

```bash
# Single level settings
export QDMPY_LOGGING__LOG_LEVEL=DEBUG

# Nested settings
export QDMPY_FIT__ESTIMATOR=LSE
export QDMPY_FIT__MAX_NUMBER_ITERATIONS=500
export QDMPY_FIT__TOLERANCE=1e-8

# Complex nested settings
export QDMPY_MODEL__CONSTRAINTS__CONSTRAINT_UNITS=mt
export QDMPY_MODEL__CONSTRAINTS__CENTER_MAX_MT=7.0
```

Then import and use:

```python
from qdmpy import get_settings

SETTINGS = get_settings()

# Environment variables are automatically loaded
print(SETTINGS.logging.log_level)      # DEBUG
print(SETTINGS.fit.estimator)          # LSE
print(SETTINGS.fit.max_number_iterations)  # 500
```

### 3. User Configuration File (Lowest Priority)

Create `~/.config/QDMpy/settings.toml` with custom settings:

```toml
[fit]
estimator = "LSE"
max_number_iterations = 500
tolerance = 1e-8

[logging]
log_level = "DEBUG"

[model.constraints]
constraint_units = "mt"
center_max_mt = 7.0
width_max_mt = 0.7
center_type = "LOWER_UPPER"

[odmr]
norm_method = "mean"

[outlier_detection]
method = "StatisticsPercentile"

[outlier_detection.StatisticsPercentile]
chi2_percentile = [0, 99.5]
width_percentile = [0, 99.0]
contrast_percentile = [1, 100]
```

Then use:

```python
from qdmpy import get_settings

SETTINGS = get_settings()

# Settings are automatically loaded from ~/.config/QDMpy/settings.toml
print(SETTINGS.fit.estimator)          # LSE
print(SETTINGS.logging.log_level)      # DEBUG
```

### Priority Order

When loading settings, QDMpy applies them in this order (highest to lowest priority):

```
Programmatic Overrides > Environment Variables > User TOML File > Defaults
```

**Example**: If you set an environment variable AND have a TOML file with different values, the environment variable wins:

```python
# ~/.config/QDMpy/settings.toml contains: estimator = "LSE"
# Environment: QDMPY_FIT__ESTIMATOR=MLE

from qdmpy import get_settings

SETTINGS = get_settings()
print(SETTINGS.fit.estimator)  # Output: MLE (environment variable wins)
```

---

## Default Values

All settings have sensible defaults built into the Pydantic models. You only need to override values that differ from your needs.

### Model Defaults

```python
from qdmpy.settings import QDMpySettings

settings = QDMpySettings()  # Uses all defaults

# Fit settings defaults
print(settings.fit.estimator)              # 'MLE'
print(settings.fit.max_number_iterations)  # 1000
print(settings.fit.tolerance)              # 1e-10

# Logging defaults
print(settings.logging.log_level)          # 'INFO'

# Model constraints defaults
constraints = settings.model.constraints
print(constraints.constraint_units)        # 'mt'
print(constraints.center_max_mt)           # 7.0
print(constraints.width_max_mt)            # 0.7
print(constraints.center_type)             # 'LOWER_UPPER'
```

### Modifying Defaults in Code

```python
from qdmpy.settings import QDMpySettings, FitSettings, LoggingSettings

# Create settings with modified defaults
settings = QDMpySettings(
    fit=FitSettings(
        estimator='LSE',  # Override default
        # max_number_iterations uses default (1000)
    ),
    logging=LoggingSettings(
        log_level='DEBUG',  # Override default
    ),
)

print(settings.fit.estimator)              # 'LSE'
print(settings.fit.max_number_iterations)  # 1000 (default)
print(settings.logging.log_level)          # 'DEBUG'
```

---

## Usage Examples

### Example 1: Basic Usage

Use the global `SETTINGS` object:

```python
from qdmpy import get_settings

SETTINGS = get_settings()

# Access settings via attribute notation
print(f"Using {SETTINGS.fit.estimator} estimator")
print(f"Max iterations: {SETTINGS.fit.max_number_iterations}")

# Settings are automatically loaded from:
# 1. Environment variables (QDMPY_*)
# 2. User config file (~/.config/QDMpy/settings.toml)
# 3. Pydantic defaults
```

### Example 2: Custom Settings for Testing

```python
from qdmpy.settings import QDMpySettings, FitSettings, ModelConstraintsSettings, ModelSettings

# Create custom settings for a test
test_settings = QDMpySettings(
    fit=FitSettings(
        estimator='LSE',
        max_number_iterations=100,
        tolerance=1e-6,
    ),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            constraint_units='absolute_ghz',
            center_min=2.80,
            center_max=2.90,
            center_type='FREE',
        )
    ),
)

# Pass test_settings directly to FitManager / Measurement APIs in tests
```

### Example 3: Environment Variable Configuration

```bash
# Run script with custom settings via environment variables
QDMPY_FIT__ESTIMATOR=LSE \
QDMPY_FIT__MAX_NUMBER_ITERATIONS=200 \
QDMPY_LOGGING__LOG_LEVEL=DEBUG \
python your_script.py
```

```python
# your_script.py
from qdmpy import get_settings

SETTINGS = get_settings()

print(f"Estimator: {SETTINGS.fit.estimator}")              # LSE
print(f"Iterations: {SETTINGS.fit.max_number_iterations}") # 200
print(f"Log Level: {SETTINGS.logging.log_level}")          # DEBUG
```

### Example 4: TOML Configuration with Partial Overrides

```toml
# ~/.config/QDMpy/settings.toml
[fit]
estimator = "LSE"
max_number_iterations = 500

[logging]
log_level = "DEBUG"

# All other settings use defaults
```

```python
from qdmpy import get_settings

SETTINGS = get_settings()

# Custom values from TOML
print(SETTINGS.fit.estimator)              # LSE
print(SETTINGS.fit.max_number_iterations)  # 500
print(SETTINGS.logging.log_level)          # DEBUG

# Default values (not specified in TOML)
print(SETTINGS.fit.tolerance)              # 1e-10 (default)
print(SETTINGS.odmr.norm_method)           # mean (default)
```

### Example 5: Accessing Constraint Settings

```python
from qdmpy import get_settings

SETTINGS = get_settings()

# Navigate the settings hierarchy
constraints = SETTINGS.model.constraints

# Access individual constraint values
print(f"Units mode: {constraints.constraint_units}")
print(f"Center max (mT): {constraints.center_max_mt}")
print(f"Width max (mT): {constraints.width_max_mt}")
print(f"Contrast: [{constraints.contrast_min}, {constraints.contrast_max}]")
print(f"Offset: [{constraints.offset_min}, {constraints.offset_max}]")

# Check constraint types
print(f"Center type: {constraints.center_type}")      # LOWER_UPPER
print(f"Width type: {constraints.width_type}")        # LOWER_UPPER
print(f"Contrast type: {constraints.contrast_type}")  # LOWER
print(f"Offset type: {constraints.offset_type}")      # FREE
```

---

## Best Practices

### 1. Use the Global SETTINGS Object

✓ **Good:**
```python
from qdmpy import get_settings

SETTINGS = get_settings()

def process_data():
    estimator = SETTINGS.fit.estimator
    # Use estimator...
```

✗ **Avoid:**
```python
from qdmpy.settings import QDMpySettings

# Creating a new instance loses environment variable overrides
custom_settings = QDMpySettings()
estimator = custom_settings.fit.estimator
```

### 2. Use Attribute Access

✓ **Good:**
```python
estimator = SETTINGS.fit.estimator
iterations = SETTINGS.fit.max_number_iterations
```

✗ **Avoid:**
```python
# Old dict-style access (pre-Pydantic)
estimator = SETTINGS['fit']['estimator']
iterations = SETTINGS['fit']['max_number_iterations']
```

### 3. Validate Settings in Your Code

```python
from qdmpy import get_settings

SETTINGS = get_settings()

# Check if configuration is valid for your use case
if SETTINGS.fit.estimator not in ['LSE', 'MLE']:
    raise ValueError(f"Invalid estimator: {SETTINGS.fit.estimator}")

if SETTINGS.fit.max_number_iterations < 100:
    print("Warning: very low iteration limit")
```

### 4. Document Configuration Requirements

```python
def fit_model(data, frequencies):
    """Fit ODMR data using configured estimator and constraints.

    Configuration requirements:
    - SETTINGS.fit.estimator: 'LSE' or 'MLE'
    - SETTINGS.fit.max_number_iterations: >= 100
    - SETTINGS.model.constraints: Valid constraint bounds

    Environment variables:
    - QDMPY_FIT__ESTIMATOR
    - QDMPY_FIT__MAX_NUMBER_ITERATIONS
    - QDMPY_MODEL__CONSTRAINTS__*
    """
    # Implementation...
```

### 5. Use Separate Configs for Different Environments

**Development (~/.config/QDMpy/settings.toml):**
```toml
[logging]
log_level = "DEBUG"

[fit]
max_number_iterations = 100  # Faster for testing
```

**Production (environment variables):**
```bash
export QDMPY_LOGGING__LOG_LEVEL=WARNING
export QDMPY_FIT__MAX_NUMBER_ITERATIONS=1000
```

### 6. Reset to Defaults

```python
from qdmpy import reset_settings

# Re-read settings from environment/TOML on next get_settings()
reset_settings()

# Next time SETTINGS is imported, it will use only defaults
# and environment variables
```

---

## Common Configuration Scenarios

### Scenario 1: Development with Verbose Logging

```toml
# ~/.config/QDMpy/settings.toml
[logging]
log_level = "DEBUG"
```

### Scenario 2: Fast Testing

```toml
# ~/.config/QDMpy/settings.toml
[fit]
max_number_iterations = 100
tolerance = 1e-6
```

### Scenario 3: Production with Conservative Fitting

```toml
# ~/.config/QDMpy/settings.toml
[fit]
estimator = "MLE"
max_number_iterations = 2000
tolerance = 1e-12

[logging]
log_level = "WARNING"
```

### Scenario 4: Custom Constraint Bounds

```toml
# ~/.config/QDMpy/settings.toml
[model.constraints]
constraint_units = "absolute_ghz"
center_min = 2.80
center_max = 2.95
center_type = "LOWER_UPPER"

width_min = 0.001
width_max = 0.01
width_type = "LOWER_UPPER"

contrast_min = 0.01
contrast_max = 0.5
contrast_type = "LOWER_UPPER"

offset_min = -0.2
offset_max = 0.2
offset_type = "LOWER_UPPER"
```

---

## Troubleshooting

### Issue: Settings Not Updating from TOML File

**Cause**: The TOML file must exist at `~/.config/QDMpy/settings.toml` and must be parsed correctly.

**Solution**:
```python
from pathlib import Path
config_file = Path.home() / '.config' / 'QDMpy' / 'settings.toml'
print(f"Config file exists: {config_file.exists()}")
print(f"Config file path: {config_file}")

# Check TOML syntax
import tomllib
with open(config_file, 'rb') as f:
    config = tomllib.load(f)
    print(config)
```

### Issue: Environment Variables Not Being Applied

**Cause**: Environment variables must use the correct prefix and nesting format.

**Solution**:
```bash
# Correct format: QDMPY_<SECTION>__<SUBSECTION>__<PARAMETER>
export QDMPY_FIT__ESTIMATOR=LSE
export QDMPY_MODEL__CONSTRAINTS__CENTER_MAX_MT=7.0

# Then verify in Python
from qdmpy import get_settings

SETTINGS = get_settings()
print(SETTINGS.fit.estimator)
print(SETTINGS.model.constraints.center_max_mt)
```

### Issue: Settings Show Defaults Instead of Custom Values

**Cause**: Configuration priority order (programmatic > env > TOML > defaults).

**Solution**:
```python
from qdmpy import get_settings

SETTINGS = get_settings()
import os

# Check what's overriding your TOML settings
print("Environment variables:")
for key, value in os.environ.items():
    if key.startswith('QDMPY_'):
        print(f"  {key}={value}")

print(f"\nConfig file: {Path.home() / '.config' / 'QDMpy' / 'settings.toml'}")
print(f"Exists: {(Path.home() / '.config' / 'QDMpy' / 'settings.toml').exists()}")
```

---

## Summary

QDMpy's Pydantic-based settings system provides:

- **Type Safety**: All settings are validated against their types
- **Flexibility**: Configure via code, environment, or files
- **Hierarchy**: Clear organization of related settings
- **IDE Support**: Autocomplete and type hints for all settings
- **Sensible Defaults**: Works out of the box without configuration

For most use cases, the global `SETTINGS` object with default values is sufficient. When customization is needed, use environment variables for deployment or TOML files for persistent configuration.
