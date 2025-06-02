from __future__ import annotations

"""
Tool registry for Saplings.

This module provides functionality for registering and managing tools that can be used by agents.
"""

import functools
import logging
from typing import Callable

from saplings._internal.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing tools that can be used by agents.

    This class provides methods for registering, retrieving, and managing tools.
    """

    def __init__(self) -> None:
        """Initialize a tool registry."""
        self.tools: dict[str, Tool] = {}

    def register(self, func_or_tool: Callable | Tool) -> None:
        """
        Register a function or Tool instance.

        Args:
        ----
            func_or_tool: Function or Tool instance to register

        """
        if isinstance(func_or_tool, Tool):
            tool = func_or_tool
        else:
            # If it's not a Tool instance, try to wrap it
            try:
                tool = Tool(
                    func=func_or_tool,
                    name=getattr(func_or_tool, "__name__", "tool"),
                    description=getattr(func_or_tool, "__doc__", "No description provided"),
                )
            except Exception as e:
                logger.exception(f"Failed to register tool {func_or_tool}: {e}")
                msg = f"Failed to register tool: {e}"
                raise ValueError(msg)

        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.

        Args:
        ----
            name: Name of the tool to get

        Returns:
        -------
            Tool or None: The tool if found, None otherwise

        """
        return self.tools.get(name)

    def list(self):
        """
        List all registered tool names.

        Returns
        -------
            List[str]: List of tool names

        """
        return list(self.tools.keys())

    def to_openai_functions(self):
        """
        Convert all tools to OpenAI function specifications.

        Returns
        -------
            List[Dict]: List of OpenAI function specifications

        """
        return [tool.to_openai_function() for tool in self.tools.values()]

    def __len__(self):
        """Get the number of registered tools."""
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool with the given name is registered."""
        return name in self.tools


# Registry to store all registered tools
_TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(name: str | None = None, description: str | None = None) -> Callable:
    """
    Decorator to register a function as a tool.

    Args:
    ----
        name: The name of the tool (defaults to the function name)
        description: A description of what the tool does

    Returns:
    -------
        Callable: Decorator function

    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool = Tool(func, name=tool_name, description=description)
        _TOOL_REGISTRY[tool_name] = tool

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_registered_tools():
    """
    Get all registered tools.

    Returns
    -------
        Dict[str, Tool]: Dictionary of registered tools

    """
    return _TOOL_REGISTRY.copy()


# We don't need to import the tool decorator here
# This was causing a circular import
# The tool decorator imports from this module, not the other way around
