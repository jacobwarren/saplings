"""
Tests to verify that public API modules only import from other public modules or their own internal modules.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Tuple

import pytest


def get_imports(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract import statements from a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        Tuple of (import statements, from import statements)
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content)
    imports = []
    from_imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                from_imports.append(node.module)

    return imports, from_imports


def is_internal_import(import_name: str) -> bool:
    """
    Check if an import is from an internal module.

    Args:
        import_name: Import name to check

    Returns:
        True if the import is from an internal module, False otherwise
    """
    # Check for _internal in the import path
    if "_internal" in import_name:
        return True

    # Check for modules with underscore prefix
    parts = import_name.split(".")
    for part in parts[1:]:  # Skip the first part (saplings)
        if part.startswith("_") and part != "_typing":
            return True

    return False


def get_module_name(file_path: str) -> str:
    """
    Get the module name from a file path.

    Args:
        file_path: Path to the Python file

    Returns:
        Module name
    """
    # Convert path to module name
    rel_path = os.path.relpath(file_path, "src")
    module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
    return module_name


def get_public_api_files() -> List[str]:
    """
    Get all public API files.

    Returns:
        List of file paths
    """
    api_dir = Path("src/saplings/api")
    return [str(f) for f in api_dir.glob("**/*.py") if not str(f).endswith("__init__.py")]


def get_component_init_files() -> List[str]:
    """
    Get all component __init__.py files.

    Returns:
        List of file paths
    """
    src_dir = Path("src/saplings")
    return [
        str(f)
        for f in src_dir.glob("*/__init__.py")
        if not str(f).startswith("src/saplings/_") and not str(f).startswith("src/saplings/api/")
    ]


def test_public_api_imports():
    """Test that public API modules only import from other public modules or their own internal modules."""
    public_api_files = get_public_api_files()

    for file_path in public_api_files:
        module_name = get_module_name(file_path)
        component_name = module_name.split(".")[1]  # e.g., "agent" from "saplings.api.agent"

        imports, from_imports = get_imports(file_path)
        all_imports = imports + from_imports

        for import_name in all_imports:
            if import_name.startswith("saplings."):
                # Skip imports from the same component's internal module
                if import_name.startswith(f"saplings.{component_name}._internal"):
                    continue

                # Skip imports from _typing
                if import_name.startswith("saplings._typing"):
                    continue

                # Check if the import is from an internal module
                if is_internal_import(import_name):
                    pytest.fail(
                        f"Public API module {module_name} imports from internal module {import_name}"
                    )


def test_component_init_imports():
    """Test that component __init__.py files only import from public modules or their own internal modules."""
    component_init_files = get_component_init_files()

    for file_path in component_init_files:
        module_name = get_module_name(file_path)
        component_name = module_name.split(".")[1]  # e.g., "agent" from "saplings.agent"

        imports, from_imports = get_imports(file_path)
        all_imports = imports + from_imports

        for import_name in all_imports:
            if import_name.startswith("saplings."):
                # Skip imports from the same component's internal module
                if import_name.startswith(f"saplings.{component_name}._internal"):
                    continue

                # Skip imports from the public API
                if import_name.startswith("saplings.api"):
                    continue

                # Skip imports from _typing
                if import_name.startswith("saplings._typing"):
                    continue

                # Check if the import is from an internal module
                if is_internal_import(import_name):
                    pytest.fail(
                        f"Component __init__.py {module_name} imports from internal module {import_name}"
                    )


if __name__ == "__main__":
    # Run the tests manually
    test_public_api_imports()
    test_component_init_imports()
    print("All tests passed!")
