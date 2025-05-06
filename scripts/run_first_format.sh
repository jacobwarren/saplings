#!/bin/bash
# Script to run the initial formatting and linting pass on the codebase
# This script performs the first step in the migration plan

set -e  # Exit on any error

echo "Running initial code formatting and linting with ruff..."
echo "This will make significant changes to the codebase to conform to the new standards."
echo "These changes should be committed as a single large PR to establish the baseline."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Run ruff formatting
echo "Running ruff formatter..."
ruff format src tests

# Run ruff linting with fixes
echo "Running ruff linter with auto-fixes..."
ruff check --fix src tests

echo ""
echo "Initial formatting complete!"
echo "Next steps:"
echo "1. Review the changes"
echo "2. Commit them as a single large PR"
echo "3. Proceed with the Week 1 adoption plan in docs/migration/quality_checks_adoption.md"
