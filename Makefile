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
EXAM_NAME = "Certified Kubernetes Administrator"

# The path to the text file containing topics, one per line.
# Example: Create a file named 'k8s_topics.txt' with contents like:
# Services & Networking
# Pods
# Storage
TOPIC_FILE = "k8s_topics.txt"

# A short base name for all generated files (e.g., 'k8s_exam').
BASE_NAME = "k8s_exam"

# Number of questions to generate per topic from the TOPIC_FILE.
QUESTIONS_PER_TOPIC = 5

# --- AUTOMATIC VARIABLES (Do not edit) ---

# The Python interpreter to use. Assumes `examexam` is in the environment.
PYTHON = python3 -m
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
		--exam-name "$(EXAM_NAME)" \
		--toc-file "$(TOPIC_FILE)" \
		--output-file "$(QUESTION_FILE)" \
		-n $(QUESTIONS_PER_TOPIC)

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
take: $(QUESTION_FILE)
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


publish: test
	rm -rf dist && hatch build