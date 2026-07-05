# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- OpenRouter as a supported model provider (alongside OpenAI, Anthropic, Google)

### Changed

- Validation now runs deterministic checks (answer-count vs. stem wording, banned option
  patterns, duplicate options/questions) before any LLM call, and records results separately
  from the LLM's good/bad verdict
- Validation LLM calls (`ask_llm`, `ask_if_bad_question`) now use structured TOML responses
  with retry-on-parse-failure instead of fragile string parsing
- A validation response that can't be parsed after retries is now recorded as "inconclusive"
  and excluded from the score, instead of silently counting as a wrong answer
- `Router` now raises `FatalConversationError` instead of calling `sys.exit()`, so a bad
  provider/model no longer kills the whole process
- Removed the non-functional Bedrock/Mistral/Cohere/Meta/AI21 mock provider, which
  previously fabricated exam content silently instead of calling a real API

### Fixed

- Exam scoring now compares selected options by identity instead of text, fixing undefined
  behavior when two options in a question share identical text
- `generate` no longer risks corrupting the output file on a crash mid-write (atomic
  write via temp file + rename)
- Corrected the `openai` dependency floor, which previously allowed a pre-1.0 install
  incompatible with the v1+ client API actually used

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

[0.1.1]: https://github.com/matthewdeanmartin/examexam/releases/tag/v0.1.1
[0.1.2]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.1...v0.1.2
[0.1.3]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.2...v0.1.3
[0.1.4]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.3...v0.1.4
[0.1.7]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.4...v0.1.7
[0.1.8]: https://github.com/matthewdeanmartin/examexam/compare/v0.1.7...v0.1.8
