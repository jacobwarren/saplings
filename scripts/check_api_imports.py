#!/usr/bin/env python
"""
Script to check API imports in the Saplings library.

This script checks that modules import from the public API.
"""

from __future__ import annotations

import sys


def check_module_imports_from_public_api(module_name: str, api_module: str) -> bool:
    """
    Check that a module imports from the public API.

    Args:
    ----
        module_name: The name of the module to check
        api_module: The name of the API module it should import from

    Returns:
    -------
        bool: True if the module imports from the public API, False otherwise

    """
    try:
        # Get the module path
        module_path = f"src/{module_name.replace('.', '/')}/__init__.py"

        # Read the source code directly from the file
        with open(module_path) as f:
            source_code = f.read()

        # Check that the module imports from the public API
        imports_from_public_api = f"from saplings.api.{api_module} import" in source_code

        # Check that it doesn't import directly from _internal
        doesnt_import_from_internal = (
            f"from {module_name}._internal" not in source_code
            or "except ImportError" in source_code
        )

        return imports_from_public_api and doesnt_import_from_internal
    except Exception as e:
        print(f"Error checking {module_name}: {e}")
        return False


def main():
    """Main function."""
    modules_to_check = [
        ("saplings.adapters", "models"),
        ("saplings.tools", "tools"),
        ("saplings.memory", "memory"),
        ("saplings.services", "services"),
        ("saplings.validator", "validator"),
        ("saplings.retrieval", "retrieval"),
        ("saplings.monitoring", "monitoring"),
        ("saplings.judge", "judge"),
    ]

    success = True

    for module_name, api_module in modules_to_check:
        result = check_module_imports_from_public_api(module_name, api_module)
        if result:
            print(f"✅ {module_name} imports from saplings.api.{api_module}")
        else:
            print(f"❌ {module_name} does not import from saplings.api.{api_module}")
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
