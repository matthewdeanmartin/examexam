#!/usr/bin/env bash
set -euo pipefail
git2md examexam \
  --ignore __init__.py __pycache__ \
  __about__.py logging_config.py py.typed \
  --output SOURCE.md