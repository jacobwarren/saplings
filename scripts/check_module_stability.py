#!/usr/bin/env python
"""
Script to check stability annotations in a specific module.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple


def check_module_for_stability_annotations(module_path: str) -> Tuple[List[str], List[str]]:
    """
    Check a module for classes with and without stability annotations.

    Args:
    ----
        module_path: The path to the module to check

    Returns:
    -------
        Tuple[List[str], List[str]]: Lists of classes with and without stability annotations

    """
    try:
        # Add the src directory to the path if it's not already there
        src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Read the file content
        with open(module_path) as f:
            content = f.read()

        # Simple parsing to find class definitions and stability annotations
        with_annotations = []
        without_annotations = []

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("class ") and i > 0:
                class_name = line.strip().split("class ")[1].split("(")[0].strip(":")
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
        print(f"Error checking {module_path}: {e}")
        return [], []


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python check_module_stability.py <module_path>")
        return 1

    module_path = sys.argv[1]

    with_annotations, without_annotations = check_module_for_stability_annotations(module_path)

    if with_annotations:
        print(f"Classes with stability annotations ({len(with_annotations)}):")
        for cls in with_annotations:
            print(f"  ✅ {cls}")

    if without_annotations:
        print(f"Classes without stability annotations ({len(without_annotations)}):")
        for cls in without_annotations:
            print(f"  ❌ {cls}")
        return 1
    elif not with_annotations:
        print("  No classes found in this module.")
    else:
        print("  All classes have stability annotations.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
