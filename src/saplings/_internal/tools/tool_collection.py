from __future__ import annotations

"""
Tool collection for Saplings.

This module provides a way to group tools together and load them from external sources.
"""


import importlib
import json
import sys
from pathlib import Path
from typing import Any

from saplings._internal.tools.base import Tool
from saplings._internal.tools.tool_registry import get_registered_tools


class ToolCollection:
    """
    A collection of tools that can be loaded from various sources.

    Tool collections provide a way to group related tools together and load them
    from external sources like directories, packages, or remote repositories.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        """
        Initialize a tool collection.

        Args:
        ----
            tools: Optional list of tools to include in the collection

        """
        self.tools = tools or []

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the collection.

        Args:
        ----
            tool: The tool to add

        """
        self.tools.append(tool)

    def get_tool(self, name: str) -> Tool | None:
        """
        Get a tool by name.

        Args:
        ----
            name: The name of the tool to get

        Returns:
        -------
            The tool if found, None otherwise

        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def get_tools(self):
        """
        Get all tools in the collection.

        Returns
        -------
            List of tools

        """
        return self.tools

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the collection to a dictionary.

        Returns
        -------
            Dictionary representation of the collection

        """
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "output_type": tool.output_type,
                }
                for tool in self.tools
            ]
        }

    def save(self, path: str | Path) -> None:
        """
        Save the collection to a file.

        Args:
        ----
            path: Path to save the collection to

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ToolCollection":
        """
        Load tools from Python files in a directory.

        Args:
        ----
            directory: Directory containing Python files with tool definitions

        Returns:
        -------
            A tool collection containing the loaded tools

        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            msg = f"Directory {directory} does not exist or is not a directory"
            raise ValueError(msg)

        # Add directory to Python path temporarily
        sys.path.insert(0, str(directory.parent))

        tools = []
        try:
            # Import all Python files in the directory
            for file_path in directory.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue

                module_name = file_path.stem
                try:
                    module = importlib.import_module(f"{directory.name}.{module_name}")

                    # Find all Tool instances in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, Tool):
                            tools.append(attr)
                except ImportError:
                    pass
        finally:
            # Remove directory from Python path
            sys.path.pop(0)

        return cls(tools)

    @classmethod
    def from_package(cls, package_name: str) -> "ToolCollection":
        """
        Load tools from a Python package.

        Args:
        ----
            package_name: Name of the package to load tools from

        Returns:
        -------
            A tool collection containing the loaded tools

        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            msg = f"Package {package_name} not found"
            raise ValueError(msg)

        tools = []

        # Find all Tool instances in the package
        for attr_name in dir(package):
            attr = getattr(package, attr_name)
            if isinstance(attr, Tool):
                tools.append(attr)

        return cls(tools)

    @classmethod
    def from_registered_tools(cls):
        """
        Create a collection from all registered tools.

        Returns
        -------
            A tool collection containing all registered tools

        """
        return cls(list(get_registered_tools().values()))
