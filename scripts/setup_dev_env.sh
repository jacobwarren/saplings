#!/bin/bash
# Script to set up the development environment with quality checks

set -e  # Exit on any error

echo "Setting up the development environment with quality checks..."

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Run initial formatting
echo "Running initial formatting with ruff..."
ruff format src tests
ruff check --fix src tests

echo "Setup complete! The following quality checks are now enabled:"
echo "- Ruff for linting and formatting"
echo "- MyPy for type checking"
echo "- Radon for cyclomatic complexity checking"
echo ""
echo "See docs/migration/quality_checks_adoption.md for more details."
