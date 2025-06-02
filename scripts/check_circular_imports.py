#!/usr/bin/env python
"""
Script to check for circular imports in the Saplings codebase.

This script analyzes the import structure of the Saplings codebase and
identifies potential circular imports.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set


def find_python_files(root_dir: str) -> List[str]:
    """
    Find all Python files in a directory.

    Args:
    ----
        root_dir: Root directory to search

    Returns:
    -------
        List of Python file paths

    """
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_imports(file_path: str) -> List[str]:
    """
    Extract imports from a Python file.

    Args:
    ----
        file_path: Path to the Python file

    Returns:
    -------
        List of imported modules

    """
    imports = []
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Extract imports
    import_pattern = re.compile(
        r"^\s*(?:from\s+(\S+)\s+import|\s*import\s+([^,\s]+))", re.MULTILINE
    )
    for match in import_pattern.finditer(content):
        if match.group(1):  # from X import Y
            imports.append(match.group(1))
        elif match.group(2):  # import X
            imports.append(match.group(2))

    return imports


def get_module_name(file_path: str, root_dir: str) -> str:
    """
    Get the module name from a file path.

    Args:
    ----
        file_path: Path to the Python file
        root_dir: Root directory of the codebase

    Returns:
    -------
        Module name

    """
    rel_path = os.path.relpath(file_path, root_dir)
    if rel_path.endswith("__init__.py"):
        rel_path = os.path.dirname(rel_path)
    else:
        rel_path = os.path.splitext(rel_path)[0]

    return rel_path.replace(os.path.sep, ".")


def build_import_graph(root_dir: str) -> Dict[str, Set[str]]:
    """
    Build a graph of imports.

    Args:
    ----
        root_dir: Root directory of the codebase

    Returns:
    -------
        Dictionary mapping modules to their imports

    """
    import_graph = defaultdict(set)
    python_files = find_python_files(root_dir)

    for file_path in python_files:
        module_name = get_module_name(file_path, root_dir)
        imports = extract_imports(file_path)

        # Filter imports to only include modules from the codebase
        for imp in imports:
            if imp.startswith("saplings"):
                import_graph[module_name].add(imp)

    return import_graph


def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Find cycles in the import graph.

    Args:
    ----
        graph: Import graph

    Returns:
    -------
        List of cycles

    """
    cycles = []
    visited = set()
    path = []

    def dfs(node: str) -> None:
        if node in path:
            # Found a cycle
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            dfs(neighbor)

        path.pop()

    for node in graph:
        dfs(node)

    return cycles


def find_circular_imports(root_dir: str) -> List[List[str]]:
    """
    Find circular imports in the codebase.

    Args:
    ----
        root_dir: Root directory of the codebase

    Returns:
    -------
        List of circular import chains

    """
    import_graph = build_import_graph(root_dir)
    cycles = find_cycles(import_graph)
    return cycles


def main():
    """Main function."""
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"

    print(f"Checking for circular imports in {root_dir}...")
    cycles = find_circular_imports(root_dir)

    if not cycles:
        print("No circular imports found.")
        return 0

    print(f"Found {len(cycles)} circular import chains:")
    for i, cycle in enumerate(cycles, 1):
        print(f"\nCircular import chain {i}:")
        for j, module in enumerate(cycle):
            if j < len(cycle) - 1:
                print(f"  {module} imports {cycle[j+1]}")
            else:
                print(f"  {module} imports {cycle[0]}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
