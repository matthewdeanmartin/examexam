# Changelog

The canonical changelog lives in the repository root at `Changelog.md`.

Recent highlights from the unreleased work:

- OpenRouter support was added as a model provider.
- Validation now does deterministic question-form checks before any LLM call.
- Structured TOML parsing and retry logic were tightened for validation responses.
- Inconclusive validation responses are recorded separately instead of being scored as wrong.
- Generation now writes question-bank updates atomically.

For the full release history, read the source file in the repository.
