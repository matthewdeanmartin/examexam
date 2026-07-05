# Contributing

This page is the generic contributor workflow: setup, git, checks, and docs commands. For architecture and internals, use [Hacking On Examexam](how-it-works.md).

## Setup

```bash
git clone https://github.com/matthewdeanmartin/examexam.git
cd examexam
uv sync --group dev
uv run pre-commit install
```

## Day-to-day commands

The repo currently has both a `Justfile` and a `Makefile`.

Common choices:

```bash
just check
make check
```

Useful focused commands:

```bash
just test
make test
make lint-check
make docs-check
uv run mkdocs serve
uv run mkdocs build
```

## Git workflow

Use a normal feature branch and open a GitHub pull request.

Typical flow:

1. Create a branch.
2. Make the smallest useful change.
3. Add or update tests when behavior changes.
4. Run checks locally.
5. Push and open a PR.

## Docs workflow

This repository now uses MkDocs for the user-facing docs in `docs/`.

Preview locally:

```bash
uv run mkdocs serve
```

Build the site:

```bash
uv run mkdocs build
```

Read the Docs uses:

- `mkdocs.yml`
- `.readthedocs.yaml`
- `docs/requirements.txt`

## Testing and quality

The project already has targets for:

- formatting
- linting
- security scanning
- tests with coverage
- docstring and doctest checks
- metadata/version consistency checks

The broadest local gate is:

```bash
make check
```

For a release-style pass:

```bash
make prerelease
```

## Secrets and local configuration

Provider API keys belong in `.env`, not in committed files.

Typical examples:

```dotenv
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

The app auto-loads `.env` on startup.
