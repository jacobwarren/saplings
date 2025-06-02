from __future__ import annotations

"""
Internal tools module for Saplings.

This module provides the internal implementation of tools components.
All components are organized in a way that avoids circular imports.
"""

# Import all components to make them available from this module
from saplings._internal.tools.base import Tool
from saplings._internal.tools.browser_tools import (
    BrowserManager,
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
    is_browser_tools_available,
    save_screenshot,
)
from saplings._internal.tools.default_tools import (
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
from saplings._internal.tools.mcp_tools import (
    MCPClient,
    MCPTool,
    is_mcp_available,
)
from saplings._internal.tools.tool import tool
from saplings._internal.tools.tool_collection import ToolCollection
from saplings._internal.tools.tool_registry import (
    ToolRegistry,
    get_registered_tools,
    register_tool,
)
from saplings._internal.tools.tool_validation import (
    validate_tool,
    validate_tool_attributes,
    validate_tool_parameters,
)

# Define __all__ to control what is exported
__all__ = [
    # Base tool class
    "Tool",
    # Tool collection
    "ToolCollection",
    # Tool registry
    "ToolRegistry",
    "register_tool",
    "get_registered_tools",
    "tool",
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
    # Tool validation
    "validate_tool",
    "validate_tool_attributes",
    "validate_tool_parameters",
]
