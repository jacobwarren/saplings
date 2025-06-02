#!/usr/bin/env python
"""
Script to check stability annotations in the Saplings API.

This script checks that public API components have stability annotations.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import List, Tuple


def get_api_modules() -> List[str]:
    """
    Get all API modules in the saplings.api package.

    Returns
    -------
        List[str]: List of module names

    """
    api_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "saplings", "api"
    )
    modules = []

    for filename in os.listdir(api_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            modules.append(filename[:-3])  # Remove .py extension

    return modules


def check_module_stability(module_name: str) -> Tuple[List[str], List[str]]:
    """
    Check stability annotations for a module.

    Args:
    ----
        module_name: Name of the module to check

    Returns:
    -------
        Tuple[List[str], List[str]]: Lists of classes with and without stability annotations

    """
    try:
        # Import the module
        module = importlib.import_module(f"saplings.api.{module_name}")

        with_annotations = []
        without_annotations = []

        for name, obj in inspect.getmembers(module):
            # Skip private attributes, non-classes, and imported classes
            if (
                name.startswith("_")
                or not inspect.isclass(obj)
                or not obj.__module__.startswith(f"saplings.api.{module_name}")
            ):
                continue

            # Check for stability annotation
            if hasattr(obj, "__stability__"):
                with_annotations.append(f"{name} ({obj.__stability__.value})")
            else:
                without_annotations.append(name)

        return with_annotations, without_annotations

    except Exception as e:
        print(f"Error checking {module_name}: {e}")
        return [], []


def main():
    """Main function."""
    # Add src directory to path
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Get all API modules
    modules = get_api_modules()

    # Check each module
    all_missing = {}
    all_annotated = {}

    for module in sorted(modules):
        with_annotations, without_annotations = check_module_stability(module)

        if without_annotations:
            all_missing[module] = without_annotations

        all_annotated[module] = with_annotations

    # Print results
    if all_missing:
        print("\n❌ The following classes are missing stability annotations:")
        for module, classes in all_missing.items():
            print(f"\nsaplings.api.{module}:")
            for cls in sorted(classes):
                print(f"  - {cls}")

        print("\n✅ Classes with stability annotations:")
        for module, classes in all_annotated.items():
            if classes:
                print(f"\nsaplings.api.{module}:")
                for cls in sorted(classes):
                    print(f"  - {cls}")

        return 1
    else:
        print("✅ All public API classes have stability annotations")
        return 0


if __name__ == "__main__":
    sys.exit(main())
