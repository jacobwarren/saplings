from __future__ import annotations

"""
Tools module for Saplings.

This module re-exports the public API from saplings.api.tools.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.
"""

# Import directly from internal modules to avoid circular imports
# We can't import from saplings.api.tools due to circular imports
# The public API test will need to be updated to handle this special case
from saplings.tools._internal.base import Tool
from saplings.tools._internal.registry import ToolRegistry, get_registered_tools
from saplings.tools._internal.registry.tool_registry import register_tool
from saplings.tools._internal.service import ToolCollection
from saplings.tools._internal.tool_decorator import tool

# Re-export symbols
__all__ = [
    "Tool",
    "ToolCollection",
    "ToolRegistry",
    "get_registered_tools",
    "register_tool",
    "tool",
    # Note: Other tool symbols should be imported from saplings.api.tools
]
