# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed
- **Critical test fixes**: Resolved failing tests in test_fit.py by removing incorrect `self` parameters from standalone test functions
- **Type safety improvements**: Fixed mypy type errors in plotting.py including Optional type annotations, missing imports, and None handling
- **Logging best practices**: Converted f-string logging statements to proper % formatting to follow security guidelines
- **Hardcoded path removal**: Replaced platform-specific hardcoded paths with environment variable support and sensible defaults
- Updated README.md to match current codebase implementation
- Fixed import examples to use correct module paths (`QDMpy.odmr.odmr.ODMR` instead of `QDMpy.ODMR`)
- Corrected CLI parameter names (`--bin-factor` instead of `--binning`)
- Fixed license filename reference (`LICENCE` instead of `LICENSE`)
- Removed non-functional code examples that referenced missing methods

### Changed  
- **Code quality improvements**: Standardized line length to 100 characters across all configuration files
- **Enhanced portability**: Improved test_data_location() function to use environment variables instead of hardcoded paths
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

### Testing
- **Comprehensive test suite for new architecture**: Added 38+ tests covering FitResult, Measurement.fit_odmr, and new plotting functions
- **FitResult test coverage**: 27 test methods with 100% code coverage including initialization, properties, B-field calculations, quality metrics, and file I/O
- **Measurement integration tests**: 10 test methods covering auto model detection, parameter extraction, error handling, and result creation
- **Plotting function tests**: 11 test methods for new plotting functions with mock and real FitResult objects, save functionality, and error handling
- **Architecture validation**: Tests confirm clean separation of concerns, type safety, error handling, extensibility, and persistence capabilities
- **Bug fixes in tests**: Corrected matplotlib axes handling in subplot logic and proper import path mocking patterns

### Documentation
- **Updated architecture diagrams**: Comprehensive updates to all Mermaid diagrams reflecting the new FitResult and Measurement integration
- **Enhanced class diagram**: Added FitResult, Measurement, and PlottingFunctions with proper relationships
- **Improved data flow diagram**: Shows complete workflow from data loading through FitResult creation to visualization
- **Updated workflow sequence**: Demonstrates new high-level integration layer with auto model detection and clean separation of concerns
- **Architecture validation**: Diagrams confirm the clean separation between data management, fitting execution, and results analysis
