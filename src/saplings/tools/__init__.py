from __future__ import annotations

"""
Tools module for Saplings.

This module provides functionality for registering and managing tools that can be used by agents.
It also includes a set of default tools that can be used out of the box.
"""


from .default_tools import (
    TOOL_MAPPING,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    GoogleSearchTool,
    PythonInterpreterTool,
    SpeechToTextTool,
    UserInputTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    get_all_default_tools,
    get_default_tool,
)
from .tool_collection import ToolCollection
from .tool_decorator import tool
from .tool_registry import Tool, ToolRegistry, get_registered_tools, register_tool
from .tool_validation import validate_tool_attributes, validate_tool_parameters

# Import MCP client if dependencies are available
try:
    from .mcp_client import MCPClient, MCPTool

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


def is_mcp_available():
    """
    Check if MCP tools are available.

    Returns
    -------
        True if MCP tools are available, False otherwise

    """
    return _MCP_AVAILABLE


# Import browser tools if dependencies are available
try:
    from .browser_tools import (
        ClickTool,
        ClosePopupsTool,
        GetPageTextTool,
        GoBackTool,
        GoToTool,
        ScrollTool,
        SearchTextTool,
        WaitTool,
        close_browser,
        get_browser_tools,
        initialize_browser,
        save_screenshot,
    )

    _BROWSER_TOOLS_AVAILABLE = True
except ImportError:
    _BROWSER_TOOLS_AVAILABLE = False


def is_browser_tools_available():
    """
    Check if browser tools are available.

    Returns
    -------
        True if browser tools are available, False otherwise

    """
    return _BROWSER_TOOLS_AVAILABLE


__all__ = [
    "TOOL_MAPPING",
    "DuckDuckGoSearchTool",
    "FinalAnswerTool",
    "GoogleSearchTool",
    # Default tools
    "PythonInterpreterTool",
    "SpeechToTextTool",
    # Core tool classes and functions
    "Tool",
    # Tool collection
    "ToolCollection",
    "ToolRegistry",
    "UserInputTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    "get_all_default_tools",
    "get_default_tool",
    "get_registered_tools",
    # Browser tools availability check
    "is_browser_tools_available",
    # MCP tools availability check
    "is_mcp_available",
    "register_tool",
    # Tool decorator
    "tool",
    # Tool validation
    "validate_tool_attributes",
    "validate_tool_parameters",
]

# Add MCP client to __all__ if available
if _MCP_AVAILABLE:
    __all__.extend(
        [
            # MCP client
            "MCPClient",
            "MCPTool",
        ]
    )

# Add browser tools to __all__ if available
if _BROWSER_TOOLS_AVAILABLE:
    __all__.extend(
        [
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
            # Browser tools
            "initialize_browser",
            "save_screenshot",
        ]
    )
