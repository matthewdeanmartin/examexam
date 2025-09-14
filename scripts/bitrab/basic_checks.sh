#! /bin/bash
set -eou pipefail
# Smoke test  all the tests that don't necessarily change anything
# exercises the arg parser mostly.
set -eou pipefail
echo "help..."
examexam --help
echo "compile help..."
examexam run --help
echo "version..."
examexam --version
echo "dry run run"
examexam run --dry-run
echo "done"

