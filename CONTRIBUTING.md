# Contributing to **examexam**

Thanks for your interest in improving **examexam**—a CLI to generate, validate, convert, and take practice exams. This guide is written for developers. The project is MIT‑licensed and authored by **Matthew Dean Martin (matthewdeanmartin)**.

> Contributors submit changes via GitHub Pull Requests.

---

## TL;DR – Quick Start

**Prereqs**

* Python **3.11+**
* [`uv`](https://github.com/astral-sh/uv) for fast deps (recommended)
* [`just`](https://github.com/casey/just) for task running (optional but used in this repo)
* `git`, `pipx` (optional, handy for tools)

**Clone & setup**

```bash
git clone https://github.com/<your-fork-or-origin>/examexam.git
cd examexam
uv sync --group dev   # installs runtime + dev deps
pre-commit install    # set up local pre-commit hooks
```

**Run the checks**

```bash
just check            # lint, tests, bandit, etc. (composite)
# or
just test             # unit tests with coverage gate
```

Open a branch, commit, push, and make a PR.

---

## Development workflow

### Using `just` (preferred)

This project ships a `Justfile` with common tasks:

* `just` (no args) — list tasks
* `just dependency-install` — install deps (`uv sync`)
* `just update-deps` — update deps + pre-commit
* `just clean` — remove build artifacts
* `just test` — run tests with coverage threshold from `pyproject.toml`
* `just isort` / `just black` — format imports/code
* `just pre-commit` — run all hooks on all files (auto-fixes when possible)
* `just pylint` — lint (includes `ruff check --fix`)
* `just bandit` — security scan
* `just check` — run mypy (currently skipped), tests, lint, bandit, pre-commit, tool audit
* `just check-docs` / `just make-docs` — docs checks / generate API docs
* `just check-md` / `just spell` — markdown & spelling checks
* `just package-check` — static dep hygiene (`deptry`)
* `just publish` — build the package with `hatch` after tests pass

> On Windows, using Git Bash or PowerShell works. The recipes call `uv` when no virtualenv is active.

### Without `just`

Equivalent basics:

```bash
uv sync --group dev
pre-commit install
pytest -vv --cov=examexam --cov-report=html --cov-fail-under $(python - <<'PY'
import toml;print(toml.load('pyproject.toml')['tool']['strict-build-script']['minimum_test_coverage'])
PY
)
ruff check . --fix
pylint examexam --rcfile=.pylintrc
bandit examexam -c pyproject.toml -r
```

---

## Environment & secrets (for LLM work)

Put provider keys in a local **`.env`** (not committed). The app auto‑loads it.

```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
EXAMEXAM_DEFAULT_MODEL=gpt4    # optional default
```

**Do not** hardcode keys in source or tests. Prefer fakes/mocks in tests. For local manual runs, pass a model name your environment supports (e.g., `--model gpt4`, `--model claude`, `--model gemini-1.5-pro`).

---

## Code style & quality

* **Formatting:** `black` (line length 120) + `isort`
* **Linting:** `ruff` (E/F/D/UP/B rules; see `pyproject.toml`) and `pylint` (thresholds in Justfile: 9.5 for lib/tests, 8.5 for scripts)
* **Typing:** type hints encouraged; `mypy` target exists but is currently skipped in the Justfile
* **Security:** `bandit` configured
* **Imports/docstrings:** see `tool.black`, `tool.isort`, `tool.ruff`/`pydocstyle` in `pyproject.toml`
* **Logging:** prefer `logging` over `print()` in library code

Run the auto-fixers before committing:

```bash
just isort black
just pre-commit
```

---

## Tests

* Framework: `pytest` (+ `pytest-cov`, `pytest-xdist`, `pytest-timeout`)
* Location: `tests/` (and `test/` is also discovered)
* Coverage gate: **read from** `pyproject.toml` → `[tool.strict-build-script].minimum_test_coverage` (defaults to 35)

Run:

```bash
just test
# or
pytest -vv --cov=examexam --cov-report=html --cov-fail-under 35
```

**Guidelines**

* Unit tests should not depend on live LLMs. Mock the router/clients.
* Add tests for new behavior; keep them fast and deterministic.
* If you touch CLI UX, include at least one high‑level test of the behavior.

---

## Docs & metadata

* `just check-docs` runs doc coverage/readability checks.
* `just make-docs` builds HTML docs via `pdoc` into `./docs`.
* Keep `README.md` user‑centric. This file (`CONTRIBUTING.md`) is for devs.

---

## Building & releasing

* Version lives in `pyproject.toml` under `[project].version`.
* Build locally:

```bash
just publish   # runs tests then builds with hatch
```

* For releases: bump version, update changelog if applicable, tag, and create a GitHub release. The maintainer handles PyPI publishing.

---

## Git workflow & PRs

1. Create an issue (or comment on an existing one) to align scope.
2. Fork and branch: `feat/…`, `fix/…`, or `docs/…` (Conventional Commits style appreciated).
3. Make focused changes with tests and docs as needed.
4. Run `just check` locally and fix any failures.
5. Push and open a Pull Request on GitHub.
6. Be responsive to review comments; we keep PRs small and iterative.

**PR checklist**

* [ ] Lint & format pass locally (`just pre-commit`)
* [ ] Tests pass with coverage ≥ configured minimum
* [ ] No secrets, temporary debug code, or large artifacts
* [ ] README/CLI help updated if user‑visible behavior changed

---

## Project layout & files (high level)

* **Question banks (TOML):** you can keep them in `data/` (convention), but any path works.
* **Session files:** saved automatically to `.session/<test-name>.toml` so users can resume.
* **CLI entrypoint:** installed as `examexam` via `[project.scripts]` in `pyproject.toml`.

> Internal TOML structure is an implementation detail—contributors should avoid exposing it in user docs.

---

## Code of conduct

Be kind, constructive, and professional. Assume good intent.

---

## License & attribution

* **License:** MIT
* **Author:** Matthew Dean Martin (matthewdeanmartin)
* Acknowledgements: OpenAI and Google Gemini models are commonly used in development and validation.
