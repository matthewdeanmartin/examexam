# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8] - 2025-09-14

### Fixed

- Handle models that don't generate code fences for TOML output; accept plain text if it parses as valid TOML

## [0.1.7] - 2025-09-06

### Added

- Jinja template support
- Configuration and initialization commands
- Research and study plan commands

### Fixed

- Missing Jinja dependency in previous release
- Empty wheel packaging issue
- Failure to stop on fatal errors
- Question generation failure handling

## [0.1.6] - 2025-09-06

### Fixed

- Corrupt wheel and oversized sdist from 0.1.5 release

## [0.1.5] - 2025-09-06

### Added

- `init` and `config` commands

### Fixed

- Stops if question generation fails
- Failure to stop on fatal errors
- `research` and `study-plan` commands
- Jinja templates

## [0.1.4] - 2025-08-25

### Added

- Version command
- Optional switches for generate command
- Progress meter for generate command

### Fixed

- TOML error in validate command
- Delay in update checker

## [0.1.3] - 2025-08-24

### Added

- Argcomplete support
- Update checker
- Did-you-mean feature for CLI

## [0.1.2] - 2025-08-24

### Added

- README

## [0.1.1] - 2025-08-24

### Added

- Initial publication

[0.1.8]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.4...v0.1.7
[0.1.4]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/matthewdeanmartin/examexam/releases/tag/v0.1.1
