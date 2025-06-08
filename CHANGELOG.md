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
