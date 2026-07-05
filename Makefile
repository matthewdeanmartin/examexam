# ==============================================================================
# Makefile for the Examexam Lifecycle
#
# Instructions:
# 1. Edit the "CONFIGURABLE VARIABLES" section below to match your exam topic.
# 2. Run `make` or `make all` to generate, validate, and convert the exam.
# 3. Run `make take` to start the exam.
# 4. Run `make clean` to remove all generated files.
# ==============================================================================

# --- CONFIGURABLE VARIABLES ---

# The full, official name of the exam. Used in prompts.
EXAM_NAME = "Gitlab Pipelines"

# The path to the text file containing topics, one per line.
# Example: Create a file named 'k8s_topics.txt' with contents like:
# Services & Networking
# Pods
# Storage
TOPIC_FILE = "input/gitlab.txt"

# A short base name for all generated files (e.g., 'k8s_exam').
BASE_NAME = "gitlab_pipelines"
MODEL = "openai"
# Number of questions to generate per topic from the TOPIC_FILE.
QUESTIONS_PER_TOPIC = 5

# --- AUTOMATIC VARIABLES (Do not edit) ---

# The Python interpreter to use. Assumes `examexam` is in the environment.
PYTHON = uv run python -m
PACKAGE_NAME = examexam

# Generated file names based on the BASE_NAME.
QUESTION_FILE = $(BASE_NAME).toml
MARKDOWN_FILE = $(BASE_NAME).md
HTML_FILE = $(BASE_NAME).html

# --- PHONY TARGETS (Commands that don't produce a file with the same name) ---

.PHONY: all generate validate convert take clean install

# --- LIFECYCLE TARGETS ---

# Default target: Runs the full creation and validation pipeline.
all: generate validate convert

# Step 1: Generate the initial question file from the topics.
generate:
	@echo ">>> Generating $(QUESTION_FILE) with $(QUESTIONS_PER_TOPIC) questions per topic..."
	$(PYTHON) $(PACKAGE_NAME) generate \
		--toc-file "$(TOPIC_FILE)" \
		--output-file "$(QUESTION_FILE)" \
		-n $(QUESTIONS_PER_TOPIC) \
		--model $(MODEL)

# Step 2: Validate the generated questions using an LLM.
validate: $(QUESTION_FILE)
	@echo ">>> Validating questions in $(QUESTION_FILE)..."
	$(PYTHON) $(PACKAGE_NAME) validate \
		--question-file "$(QUESTION_FILE)" \
		--exam-name "$(EXAM_NAME)"

# Step 3: Convert the validated TOML file into pretty formats.
convert: $(QUESTION_FILE)
	@echo ">>> Converting $(QUESTION_FILE) to Markdown and HTML..."
	$(PYTHON) $(PACKAGE_NAME) convert \
		--input-file "$(QUESTION_FILE)" \
		--output-base-name "$(BASE_NAME)"

# Standalone Step: Take the exam.
take:
	@echo ">>> Starting exam from $(QUESTION_FILE)..."
	$(PYTHON) $(PACKAGE_NAME) take --question-file "$(QUESTION_FILE)"

# --- UTILITY TARGETS ---

# Remove all generated files.
clean:
	@echo ">>> Cleaning up generated files..."
	rm -f $(QUESTION_FILE) $(MARKDOWN_FILE) $(HTML_FILE)

# Install the package in editable mode for development.
install:
	pip install -e .


# ==============================================================================
# Quality gates
#
# Adapted from ../do_i_need_to_upgrade's Makefile. Only targets whose tooling is
# actually declared in this project's [dependency-groups].dev are wired in.
# Invoke everything through `uv run` so it uses the locked virtualenv.
# ==============================================================================

UV ?= uv
PACKAGE := examexam
PYTHON_TARGETS := examexam tests
PYLINT_MAIN_TARGETS := examexam
PYLINT_TEST_TARGETS := tests
MARKDOWN_TARGETS := README.md CHANGELOG.md docs
ABOUT_FILE := examexam/__about__.py

.PHONY: \
	sync pre-commit-install \
	format format-python format-markdown \
	format-check format-check-python format-check-markdown \
	lint lint-check ruff-fix ruff-check pylint pylint-tests pylint-spelling \
	spell \
	docs-check docs-check-docstrings docs-check-pydoctest \
	test test-ci smoke \
	typecheck typecheck-mypy \
	security bandit \
	metadata metadata-check version-check dev-status \
	gha-validate \
	dont-be-lazy pydoc-docs \
	check check-ci prerelease prerelease-check publish-check help

help:
	@echo "Quality gate targets:"
	@echo "  format          Auto-format python + markdown"
	@echo "  format-check    Check formatting without changes"
	@echo "  lint            Ruff --fix + pylint (main + tests)"
	@echo "  lint-check      Ruff check + pylint (read-only)"
	@echo "  spell           Spell-check code and docs"
	@echo "  test            pytest with coverage"
	@echo "  test-ci         pytest -n auto (parallel, for CI)"
	@echo "  typecheck       mypy"
	@echo "  security        bandit"
	@echo "  docs-check      interrogate docstring coverage + pydoctest"
	@echo "  metadata-check  Verify __about__.py is in sync"
	@echo "  version-check   jiggle_version consistency check"
	@echo "  dev-status      Development Status classifier check"
	@echo "  gha-validate    Parse workflows + zizmor"
	@echo "  check           Full local quality gate"
	@echo "  check-ci        CI quality gate (parallel tests)"
	@echo "  prerelease      All checks before publishing"

sync:
	@$(UV) sync

pre-commit-install:
	@$(UV) run pre-commit install

# ── Formatting ────────────────────────────────────────────────────────────────

format: format-python format-markdown

format-python:
	@$(UV) run isort $(PYTHON_TARGETS)
	@$(UV) run black $(PYTHON_TARGETS)
	@$(UV) run ruff check --fix $(PYTHON_TARGETS)
	@$(UV) run black $(PYTHON_TARGETS)

format-markdown:
	@$(UV) run mdformat $(MARKDOWN_TARGETS)

format-check: format-check-python format-check-markdown

format-check-python:
#	@$(UV) run isort --check-only $(PYTHON_TARGETS)
#	@$(UV) run black --check $(PYTHON_TARGETS)
	@$(UV) run ruff check $(PYTHON_TARGETS)

format-check-markdown:
	@$(UV) run mdformat --check $(MARKDOWN_TARGETS)

# ── Linting ───────────────────────────────────────────────────────────────────

lint: ruff-fix pylint pylint-tests

lint-check: ruff-check pylint pylint-tests

ruff-fix:
	@$(UV) run ruff check --fix $(PYTHON_TARGETS)

ruff-check:
	@$(UV) run ruff check $(PYTHON_TARGETS)

pylint:
	@$(UV) run pylint --score=n --reports=n --rcfile=.pylintrc $(PYLINT_MAIN_TARGETS)

pylint-tests:
	@$(UV) run pylint --score=n --reports=n --rcfile=.pylintrc_tests $(PYLINT_TEST_TARGETS)

pylint-spelling:
	@$(UV) run pylint --score=n --reports=n --rcfile=.pylintrc_spell $(PYLINT_MAIN_TARGETS)

# ── Spell check ───────────────────────────────────────────────────────────────

spell:
	@$(UV) run codespell $(PACKAGE) tests README.md CHANGELOG.md docs

# ── Documentation checks ─────────────────────────────────────────────────────

docs-check: docs-check-docstrings docs-check-pydoctest

docs-check-docstrings:
	@$(UV) run interrogate $(PACKAGE) --verbose --fail-under 70

docs-check-pydoctest:
	@$(UV) run pydoctest --config .pydoctest.json \
		| grep -v "__init__" | grep -v "__main__" | grep -v "Unable to parse" || true

# ── Tests ─────────────────────────────────────────────────────────────────────

smoke:
	@$(UV) run examexam --version
	@$(UV) run examexam --help >/dev/null

test:
	@$(UV) run pytest -q \
		--cov=$(PACKAGE) \
		--cov-report=html \
		--junitxml=junit.xml \
		--timeout=60

test-ci:
	@$(UV) run pytest -q -n auto --dist=loadfile \
		--cov=$(PACKAGE) \
		--cov-report=xml \
		--junitxml=junit.xml \
		--timeout=60

# ── Type checking ─────────────────────────────────────────────────────────────

typecheck: typecheck-mypy

typecheck-mypy:
	@$(UV) run mypy --hide-error-context $(PACKAGE)

# ── Security ──────────────────────────────────────────────────────────────────

security: bandit

bandit:
	@$(UV) run bandit -q -r $(PACKAGE)

# ── Metadata / version ───────────────────────────────────────────────────────

metadata:
	@$(UV) run metametameta pep621 --name $(PACKAGE) --source pyproject.toml --output $(ABOUT_FILE)
	@$(UV) run isort $(ABOUT_FILE)
	@$(UV) run black $(ABOUT_FILE)

metadata-check:
	@$(UV) run metametameta sync-check --output $(ABOUT_FILE)

version-check:
	@$(UV) run jiggle_version check

dev-status:
	@$(UV) run --with troml-dev-status>=0.6.0 troml-dev-status validate .

# ── GitHub Actions maintenance ───────────────────────────────────────────────

gha-validate:
	@echo "Validating GitHub Actions workflows"
	@$(UV) run python -c "import pathlib, yaml; [yaml.safe_load(p.read_text(encoding='utf-8')) for p in pathlib.Path('.github/workflows').glob('*.yml')]; print('YAML parse OK')"
	@uvx zizmor --no-progress --no-exit-codes .

# ── Aggregate gates ───────────────────────────────────────────────────────────

check: format-check lint-check security test smoke typecheck metadata-check version-check
	@echo "All checks passed."

check-ci: lint-check security test-ci smoke typecheck metadata-check version-check
	@echo "CI checks passed."

prerelease: check dev-status docs-check spell publish-check
	@echo "Pre-release checks complete — ready to publish."

publish-check:
	@$(UV) build
	@echo "Distribution built — inspect dist/ before publishing."
	@ls -lh dist/

publish: test
	rm -rf dist && hatch build

# ── Dogfooding targets (independent, not wired into check) ───────────────────

.PHONY: prerelease-check
prerelease-check: version-check dev-status
	@echo "Pre-release checks passed."

dont-be-lazy:
	@uv run dont_be_lazy --root . --no-color summary
	@uv run dont_be_lazy --root . --no-color scan examexam --no-config-suppressions || true

pydoc-docs:
	@uv run pydoc_fork examexam -o ./pydoc/
