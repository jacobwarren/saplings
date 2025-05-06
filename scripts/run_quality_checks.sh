#!/bin/bash
# Run all standardized code quality checks
set -e

# Print section header
function section() {
  echo
  echo "======================================================================"
  echo "  $1"
  echo "======================================================================"
  echo
}

section "Installing pre-commit if needed"
if ! command -v pre-commit &> /dev/null; then
  echo "pre-commit not found, installing..."
  pip install pre-commit
else
  echo "pre-commit is already installed"
fi

section "Running pre-commit hooks"
pre-commit install
pre-commit run --all-files

echo
echo "✅ All code quality checks passed!"
echo
echo "Your code meets all the project's quality standards and is ready to commit."
