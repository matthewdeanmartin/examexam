# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

- Added for new features.
- Changed for changes in existing functionality.
- Deprecated for soon-to-be removed features.
- Removed for now removed features.
- Fixed for any bug fixes.
- Security in case of vulnerabilities.


## [0.1.8] - 2025-09-06

### Fixed

- Some models don't or won't generate toml with code fences. Now accepts text if it looks like the right toml.

## [0.1.7] - 2025-09-06

### Fixed

- 0.1.6's was missing jinja


## [0.1.6] - 2025-09-06

### Fixed

- 0.1.5's wheel was corrupt and sdist had too much junk

## [0.1.5] - 2025-09-06

### Fixed

- Stops if question generation fails
- Hack to fix code from failing to stop on fatal errors
- `research` and `study-plan` commands. 
- Jinja templates.

### Added

- init, conig


## [0.1.4] - 2025-08-24

### Fixed

- TOML error in Validate fixed
- Delay in update checker fixed.

### Added

- version
- Make more switches optional for `generate`
- progress meter for `generate`

## [0.1.3] - 2025-08-24

### Added

- argcomplete, update checker, did-you-mean feature for CLI

## [0.1.2] - 2025-08-23

### Added

- README.md

## [0.1.1] - 2025-08-23

### Added

- Initial publication