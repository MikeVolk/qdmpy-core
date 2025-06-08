# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed
- Updated README.md to match current codebase implementation
- Fixed import examples to use correct module paths (`QDMpy.odmr.odmr.ODMR` instead of `QDMpy.ODMR`)
- Corrected CLI parameter names (`--bin-factor` instead of `--binning`)
- Fixed license filename reference (`LICENCE` instead of `LICENSE`)
- Removed non-functional code examples that referenced missing methods

### Changed  
- Updated Quick Start guide with working code examples based on actual functionality
- Replaced invalid installation option `pip install QDMpy[gpu]` with proper GPU setup documentation
- Updated development setup instructions to use `uv run` commands consistently
- Enhanced CLI documentation with complete list of available options and commands
- Improved module descriptions to reflect actual package structure with `odmr` subpackage
- Added documentation for processor classes and pipeline system

### Added
- Comprehensive CLI usage examples showing all available commands
- GPU/CUDA requirements explanation with bundled pyGpufit wheel information
- Development environment verification command for GPU acceleration
- **New FitResult class**: Lightweight, data-only container for ODMR fitting results
- **Enhanced Measurement class**: Added `fit_odmr()` method with intelligent model auto-detection
- **Separated plotting functions**: Moved visualization logic from FitResult to `QDMpy.plotting` module
- Comprehensive magnetic field calculation from fitted resonance frequencies
- Quality metrics calculation and result persistence functionality
- Support for multiple fitting models (ESR14N, ESR15N, ESRSINGLE) with automatic parameter extraction

### Changed
- **Major architecture refactor**: Implemented clean separation between data management (Measurement), fitting execution (FitManager), and results analysis (FitResult)
- **Decoupled FitResult**: Removed heavy object dependencies, now stores only essential data and metadata
- **Plotting interface**: Functions now take FitResult objects as input parameters instead of being methods
- Model auto-detection in Measurement.fit_odmr() when model_name=None (removed redundant "auto" option)
- FitResult objects are now lightweight and easily serializable without object reconstruction

### Fixed
- **Type safety improvements**: Resolved mypy type checking issues in refactored components
- **Import handling**: Fixed circular import issues between plotting, measurement, and result modules  
- **Code formatting**: Applied ruff auto-fixes for 102+ formatting and style issues
- **Type annotations**: Corrected self parameter annotations and return type specifications
- **Data serialization**: Fixed np.savez_compressed compatibility issues with numpy type conversion
- **Docstring consistency**: Standardized parameter documentation and Google-style format compliance
