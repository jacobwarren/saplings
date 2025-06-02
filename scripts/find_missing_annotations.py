#!/usr/bin/env python
"""
Script to find classes in the public API that are missing stability annotations.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import List, Tuple


def check_module_for_missing_annotations(module_name: str) -> Tuple[List[str], List[str]]:
    """
    Check a module for classes without stability annotations.

    Args:
    ----
        module_name: The name of the module to check

    Returns:
    -------
        Tuple[List[str], List[str]]: Lists of classes with and without stability annotations

    """
    try:
        # Add the src directory to the path if it's not already there
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Import the module
        module = importlib.import_module(module_name)

        with_annotations = []
        without_annotations = []

        for name, obj in inspect.getmembers(module):
            # Skip private attributes and non-classes
            if name.startswith("_") or not inspect.isclass(obj):
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
    # List of API modules to check
    api_modules = [
        "saplings.api.monitoring",
        "saplings.api.judge",
        "saplings.api.models",
        "saplings.api.tools",
        "saplings.api.memory",
        "saplings.api.services",
        "saplings.api.validator",
        "saplings.api.retrieval",
        "saplings.api.gasa",
        "saplings.api.orchestration",
        "saplings.api.security",
        "saplings.api.modality",
        "saplings.api.self_heal",
        "saplings.api.tool_factory",
    ]

    missing_annotations = False

    for module_name in api_modules:
        print(f"\nChecking {module_name}...")
        try:
            with_annotations, without_annotations = check_module_for_missing_annotations(
                module_name
            )

            if with_annotations:
                print(f"Classes with stability annotations ({len(with_annotations)}):")
                for cls in with_annotations:
                    print(f"  ✅ {cls}")

            if without_annotations:
                print(f"Classes without stability annotations ({len(without_annotations)}):")
                for cls in without_annotations:
                    print(f"  ❌ {cls}")
                missing_annotations = True
            elif not with_annotations:
                print("  No classes found in this module.")
            else:
                print("  All classes have stability annotations.")
        except Exception as e:
            print(f"Error processing {module_name}: {e}")

    return 1 if missing_annotations else 0


if __name__ == "__main__":
    sys.exit(main())
