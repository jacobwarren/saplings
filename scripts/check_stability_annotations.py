#!/usr/bin/env python
"""
Script to check stability annotations in the Saplings library.

This script checks that public API components have stability annotations.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys


def check_module_stability_annotations(api_module: str) -> bool:
    """
    Check that a module's public API components have stability annotations.

    Args:
    ----
        api_module: The name of the API module to check

    Returns:
    -------
        bool: True if all public API components have stability annotations, False otherwise

    """
    try:
        # Add the src directory to the path if it's not already there
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Import the module
        module = importlib.import_module(f"saplings.api.{api_module}")

        # Check that the components have stability annotations
        missing_annotations = []

        for name, obj in inspect.getmembers(module):
            # Skip private attributes and non-classes
            if name.startswith("_") or not inspect.isclass(obj):
                continue

            # Skip imported classes that are not defined in this module
            if obj.__module__ != f"saplings.api.{api_module}":
                continue

            # Check that the class has a stability annotation
            if not hasattr(obj, "__stability__"):
                missing_annotations.append(name)

        if missing_annotations:
            print(f"❌ saplings.api.{api_module} has classes without stability annotations:")
            for name in missing_annotations:
                print(f"  - {name}")
            return False
        else:
            print(f"✅ saplings.api.{api_module} has stability annotations for all classes")
            return True
    except Exception as e:
        print(f"Error checking saplings.api.{api_module}: {e}")
        return False


def main():
    """Main function."""
    modules_to_check = [
        "models",
        "tools",
        "memory",
        "services",
        "validator",
        "retrieval",
        "monitoring",
        "judge",
    ]

    success = True

    for api_module in modules_to_check:
        result = check_module_stability_annotations(api_module)
        if not result:
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
