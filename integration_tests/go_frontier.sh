#!/usr/bin/env bash
set -euo pipefail

# User generates study plan and studies.
python -m examexam study-plan --toc-file integration_tests/cats.txt --model-class frontier
# User generates an exam
python -m examexam generate --toc integration_tests/cats.txt -n 2 --model-class frontier
# User converts to a human friendly format
python -m examexam convert --input-file integration_tests/cats.toml --output-base-name cat
rm cat.md cat.html

# Validate exam to find potential bad questions.
python -m examexam validate --question-file integration_tests/cats.toml --model-class frontier

export EXAMEXAM_NONINTERACTIVE=1
# User generates a study plan on a single topic
python -m examexam research --topic tardigraves --model-class frontier

export EXAMEXAM_MACHINE_TAKES_EXAM=1
python -m examexam take --question-file integration_tests/cats.toml

# Advanced Scenarios to allow user to modify prompts used in generation/validation
python -m examexam customize
rm -rf prompts
