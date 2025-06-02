from __future__ import annotations

"""
Internal module for tools components.

This module provides the implementation of tools components for the Saplings framework.
"""

# Import from individual modules
from saplings.tools._internal.base import Tool

# Import lightweight tools directly
from saplings.tools._internal.implementations import (
    TOOL_MAPPING,
    # Browser tools
    BrowserManager,
    ClickTool,
    ClosePopupsTool,
    # Default tools
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    GetPageTextTool,
    GoBackTool,
    GoogleSearchTool,
    GoToTool,
    # MCP tools
    MCPClient,
    MCPTool,
    PythonInterpreterTool,
    ScrollTool,
    SearchTextTool,
    UserInputTool,
    VisitWebpageTool,
    WaitTool,
    WikipediaSearchTool,
    close_browser,
    get_all_default_tools,
    get_browser_tools,
    get_default_tool,
    initialize_browser,
    is_browser_tools_available,
    is_mcp_available,
    save_screenshot,
)

# Import from subdirectories
from saplings.tools._internal.registry import (
    ToolRegistry,
    get_registered_tools,
)
from saplings.tools._internal.service import (
    ToolCollection,
)
from saplings.tools._internal.tool_decorator import tool


# Lazy import for heavy tools
def __getattr__(name: str):
    """Lazy import heavy tools to avoid loading dependencies during basic import."""
    if name == "SpeechToTextTool":
        from saplings.tools._internal.implementations import SpeechToTextTool

        return SpeechToTextTool
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core components
    "Tool",
    "tool",
    # Registry
    "ToolRegistry",
    "get_registered_tools",
    # Service
    "ToolCollection",
    # Default tools
    "TOOL_MAPPING",
    "DuckDuckGoSearchTool",
    "FinalAnswerTool",
    "GoogleSearchTool",
    "PythonInterpreterTool",
    "SpeechToTextTool",
    "UserInputTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    "get_all_default_tools",
    "get_default_tool",
    # Browser tools
    "BrowserManager",
    "ClickTool",
    "ClosePopupsTool",
    "GetPageTextTool",
    "GoBackTool",
    "GoToTool",
    "ScrollTool",
    "SearchTextTool",
    "WaitTool",
    "close_browser",
    "get_browser_tools",
    "initialize_browser",
    "is_browser_tools_available",
    "save_screenshot",
    # MCP tools
    "MCPClient",
    "MCPTool",
    "is_mcp_available",
]
