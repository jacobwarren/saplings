from __future__ import annotations

"""
Tool implementations module.

This module provides concrete tool implementations for the Saplings framework.
"""

# Import lightweight tools directly
from saplings.tools._internal.implementations.default_tools import (
    TOOL_MAPPING,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    GoogleSearchTool,
    PythonInterpreterTool,
    UserInputTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    get_all_default_tools,
    get_default_tool,
)


# Lazy import for heavy tools
def __getattr__(name: str):
    """Lazy import heavy tools to avoid loading dependencies during basic import."""
    if name == "SpeechToTextTool":
        from saplings.tools._internal.implementations.default_tools import SpeechToTextTool

        return SpeechToTextTool
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from saplings.tools._internal.implementations.browser_tools import (
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
from saplings.tools._internal.implementations.mcp_client import (
    MCPClient,
    MCPTool,
    is_mcp_available,
)

__all__ = [
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
