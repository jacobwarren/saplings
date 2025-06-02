#!/usr/bin/env python
"""
Script to check API usage in examples.

This script checks that examples use the public API.
"""

from __future__ import annotations

import os
import re
import sys


def check_file_uses_public_api(file_path: str) -> bool:
    """
    Check that a file uses the public API.

    Args:
    ----
        file_path: The path to the file to check

    Returns:
    -------
        bool: True if the file uses the public API, False otherwise

    """
    try:
        # Read the file
        with open(file_path) as f:
            content = f.read()

        # Skip examples that intentionally use internal APIs
        if os.path.basename(file_path) in [
            "secure_hot_loading_example.py",
            "gasa_test.py",
            "self_healing_monitor_with_run_updated.py",
        ]:
            print(f"⚠️ {file_path} is allowed to use internal APIs for educational purposes")
            return True

        # Check for imports from internal modules
        internal_import_pattern = r"from\s+saplings\..*\._internal"
        internal_imports = re.findall(internal_import_pattern, content)

        if internal_imports:
            print(f"❌ {file_path} imports from internal modules:")
            for imp in internal_imports:
                print(f"  - {imp}")
            return False

        # Check for imports from component modules instead of the public API
        component_import_pattern = r"from\s+saplings\.(memory|tools|services|validator|retrieval|monitoring|judge)\s+import"
        component_imports = re.findall(component_import_pattern, content)

        if component_imports:
            print(f"⚠️ {file_path} imports from component modules instead of the public API:")
            for comp in component_imports:
                print(f"  - from saplings.{comp} import ...")
            return False

        # Check for imports from the public API
        public_api_import_pattern = r"from\s+saplings(\s+import|\.(api|core)\s+import)"
        public_api_imports = re.findall(public_api_import_pattern, content)

        if public_api_imports:
            print(f"✅ {file_path} imports from the public API")
            return True

        # If no imports are found, it's not using the API at all
        print(f"⚠️ {file_path} does not import from Saplings")
        return True

    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return False


def main():
    """Main function."""
    examples_dir = "examples"

    # Get all Python files in the examples directory
    example_files = []
    for root, _, files in os.walk(examples_dir):
        for file in files:
            if file.endswith(".py"):
                example_files.append(os.path.join(root, file))

    # Check each file
    success = True
    for file in example_files:
        result = check_file_uses_public_api(file)
        if not result:
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
