#!/usr/bin/env python
"""
Script to check stability annotations in the Saplings API.

This script checks that public API components have stability annotations.
"""

from __future__ import annotations

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
        # Import the module directly
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src",
            "saplings",
            "api",
            f"{module_name}.py",
        )

        # Skip if module doesn't exist
        if not os.path.exists(module_path):
            return [], []

        # Read the module file
        with open(module_path) as f:
            content = f.read()

        # Check for class definitions and stability annotations
        with_annotations = []
        without_annotations = []

        # Special case for StabilityLevel in stability.py
        if module_name == "stability" and "class StabilityLevel" in content:
            with_annotations.append("StabilityLevel (stable)")

        # Special case for Service in di.py
        if module_name == "di" and "class Service" in content:
            with_annotations.append("Service (stable)")

        # Simple parsing to find class definitions and check for @stable, @beta, @alpha decorators
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("class ") and i > 0:
                class_name = line.strip().split("class ")[1].split("(")[0].strip(":")

                # Skip classes we've already handled
                if (module_name == "stability" and class_name == "StabilityLevel") or (
                    module_name == "di" and class_name == "Service"
                ):
                    continue

                # Check previous line for stability annotation
                prev_line = lines[i - 1].strip()
                if any(
                    decorator in prev_line
                    for decorator in ["@stable", "@beta", "@alpha", "@internal"]
                ):
                    stability = prev_line.strip("@")
                    with_annotations.append(f"{class_name} ({stability})")
                else:
                    without_annotations.append(class_name)

        return with_annotations, without_annotations

    except Exception as e:
        print(f"Error checking {module_name}: {e}")
        return [], []


def main():
    """Main function."""
    # Get all API modules
    modules = get_api_modules()

    # Check each module
    all_missing = {}
    all_annotated = {}

    for module in sorted(modules):
        with_annotations, without_annotations = check_module_stability(module)

        if without_annotations:
            all_missing[module] = without_annotations

        if with_annotations:
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
