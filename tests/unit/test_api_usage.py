from __future__ import annotations

"""
Tests for API usage in examples.

This module contains tests to ensure that examples use the public API
instead of internal APIs.
"""

import ast
from pathlib import Path
from typing import List, Tuple


def get_example_files() -> List[str]:
    """Get all Python example files."""
    examples_dir = Path("examples")
    return [str(path) for path in examples_dir.glob("*.py") if path.is_file()]


def extract_imports(file_path: str) -> List[Tuple[str, List[str]]]:
    """
    Extract imports from a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        List of tuples (module, [imported_names])
    """
    with open(file_path) as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Skip files with syntax errors
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append((name.name, [name.name]))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.append((node.module, [name.name for name in node.names]))

    return imports


def is_internal_import(module: str, name: str) -> bool:
    """
    Check if an import is from an internal module.

    Args:
        module: The module name
        name: The imported name

    Returns:
        True if the import is from an internal module, False otherwise
    """
    # Check if the module is an internal module
    if module.startswith("saplings."):
        # Check for _internal in the module path
        if "_internal" in module:
            return True
        # Check for underscore prefix in module components
        parts = module.split(".")
        for part in parts[1:]:  # Skip the first part (saplings)
            if part.startswith("_") and not part.startswith("__"):
                return True

    # Check if the name has an underscore prefix
    if name.startswith("_") and not name.startswith("__"):
        return True

    return False


def get_internal_imports(file_path: str) -> List[Tuple[str, str]]:
    """
    Get all internal imports from a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        List of tuples (module, name) for internal imports
    """
    imports = extract_imports(file_path)
    internal_imports = []

    for module, names in imports:
        for name in names:
            if is_internal_import(module, name):
                internal_imports.append((module, name))

    return internal_imports


def test_examples_use_public_api():
    """Test that examples use the public API instead of internal APIs."""
    example_files = get_example_files()
    assert len(example_files) > 0, "No example files found"

    # Collect all internal imports
    all_internal_imports = {}
    for file_path in example_files:
        internal_imports = get_internal_imports(file_path)
        if internal_imports:
            all_internal_imports[file_path] = internal_imports

    # Check if any internal imports were found
    if all_internal_imports:
        # Format the error message
        error_message = "The following examples use internal APIs:\n"
        for file_path, imports in all_internal_imports.items():
            error_message += f"\n{file_path}:\n"
            for module, name in imports:
                error_message += f"  - from {module} import {name}\n"

        # Suggest public API alternatives
        error_message += "\nSuggested public API alternatives:\n"
        error_message += "  - from saplings import Agent, AgentBuilder, AgentConfig\n"
        error_message += "  - from saplings import LLM, LLMBuilder, LLMResponse\n"
        error_message += "  - from saplings import MemoryStoreBuilder, DependencyGraphBuilder\n"
        error_message += "  - from saplings import Tool, ToolRegistry, ToolCollection\n"
        error_message += "  - from saplings import PythonInterpreterTool, WikipediaSearchTool\n"

        assert not all_internal_imports, error_message


def test_tools_api_uses_internal_imports():
    """Test that the tools API module imports from the internal tools module."""
    # Get the tools API file
    tools_api_file = Path("src/saplings/api/tools.py")

    # Check that the file exists
    if not tools_api_file.exists():
        # Skip the test if the file doesn't exist
        return

    # Get the file content
    with open(tools_api_file) as f:
        content = f.read()

    # Check for internal imports
    assert (
        "from saplings._internal.tools import" in content
    ), "Tools API does not import from internal tools module"
