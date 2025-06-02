from __future__ import annotations

"""
Tools API module for Saplings.

This module provides the public API for tools and related components.
"""

from saplings.tools._internal.base import Tool as _Tool
from saplings.tools._internal.implementations.default_tools import (
    TOOL_MAPPING as _TOOL_MAPPING,
)

# Import lightweight tools directly
from saplings.tools._internal.implementations.default_tools import (
    DuckDuckGoSearchTool as _DuckDuckGoSearchTool,
)
from saplings.tools._internal.implementations.default_tools import (
    FinalAnswerTool as _FinalAnswerTool,
)
from saplings.tools._internal.implementations.default_tools import (
    GoogleSearchTool as _GoogleSearchTool,
)
from saplings.tools._internal.implementations.default_tools import (
    PythonInterpreterTool as _PythonInterpreterTool,
)
from saplings.tools._internal.implementations.default_tools import (
    UserInputTool as _UserInputTool,
)
from saplings.tools._internal.implementations.default_tools import (
    VisitWebpageTool as _VisitWebpageTool,
)
from saplings.tools._internal.implementations.default_tools import (
    WikipediaSearchTool as _WikipediaSearchTool,
)
from saplings.tools._internal.implementations.default_tools import (
    get_all_default_tools as _get_all_default_tools,
)
from saplings.tools._internal.implementations.default_tools import (
    get_default_tool as _get_default_tool,
)
from saplings.tools._internal.registry import (
    ToolRegistry as _ToolRegistry,
)
from saplings.tools._internal.registry import (
    get_registered_tools as _get_registered_tools,
)
from saplings.tools._internal.service import ToolCollection as _ToolCollection
from saplings.tools._internal.tool_decorator import tool as _tool


# Lazy import for heavy tools
def _get_speech_to_text_tool():
    """Lazy import SpeechToTextTool to avoid loading heavy dependencies."""
    from saplings.tools._internal.implementations.default_tools import SpeechToTextTool

    return SpeechToTextTool


from saplings.api.stability import beta, stable
from saplings.tools._internal.implementations.browser_tools import (
    BrowserManager as _BrowserManager,
)
from saplings.tools._internal.implementations.browser_tools import (
    is_browser_tools_available as _is_browser_tools_available,
)
from saplings.tools._internal.implementations.mcp_client import (
    is_mcp_available as _is_mcp_available,
)
from saplings.tools._internal.registry.tool_registry import register_tool as _register_tool
from saplings.tools._internal.tool_validation import (
    validate_tool as _validate_tool,
)
from saplings.tools._internal.tool_validation import (
    validate_tool_attributes as _validate_tool_attributes,
)
from saplings.tools._internal.tool_validation import (
    validate_tool_parameters as _validate_tool_parameters,
)

# MCP tools are imported directly in the __init__.py file
# We don't need to import them here

# Browser tools are imported directly in the __init__.py file
# We don't need to import them here

# Re-export the TOOL_MAPPING constant
TOOL_MAPPING = _TOOL_MAPPING

# Define the public API
__all__ = [
    # Base tool class
    "Tool",
    # Tool collection
    "ToolCollection",
    # Tool registry
    "ToolRegistry",
    # Default tools
    "DuckDuckGoSearchTool",
    "FinalAnswerTool",
    "GoogleSearchTool",
    "PythonInterpreterTool",
    "SpeechToTextTool",
    "UserInputTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    # Tool functions
    "get_all_default_tools",
    "get_default_tool",
    "get_registered_tools",
    "is_browser_tools_available",
    "is_mcp_available",
    "register_tool",
    "tool",
    # Tool validation
    "validate_tool",
    "validate_tool_attributes",
    "validate_tool_parameters",
    # Browser tools
    "BrowserManager",
    # Constants
    "TOOL_MAPPING",
]


# Re-export the Tool class with its public API
@stable
class Tool(_Tool):
    """
    Base class for all tools.

    This class defines the interface for all tools in the Saplings framework.
    It provides methods for executing the tool and accessing its metadata.
    """


# Re-export the ToolCollection class with its public API
@stable
class ToolCollection(_ToolCollection):
    """
    Collection of related tools.

    This class provides a way to group related tools together and manage
    them as a single entity.
    """


# Re-export the ToolRegistry class with its public API
@stable
class ToolRegistry(_ToolRegistry):
    """
    Registry for managing tools.

    This class provides a way to register and retrieve tools by name.
    """


# Re-export the default tools with their public APIs
@stable
class DuckDuckGoSearchTool(_DuckDuckGoSearchTool):
    """Tool for searching the web using DuckDuckGo."""


@stable
class FinalAnswerTool(_FinalAnswerTool):
    """Tool for providing a final answer to a task."""


@stable
class GoogleSearchTool(_GoogleSearchTool):
    """Tool for searching the web using Google."""


@stable
class PythonInterpreterTool(_PythonInterpreterTool):
    """Tool for executing Python code."""


@beta
class SpeechToTextTool:
    """Tool for transcribing speech to text."""

    def __new__(cls, *args, **kwargs):
        """Create a new SpeechToTextTool instance using lazy import."""
        _SpeechToTextTool = _get_speech_to_text_tool()
        return _SpeechToTextTool(*args, **kwargs)


@stable
class UserInputTool(_UserInputTool):
    """Tool for getting input from the user."""


@stable
class VisitWebpageTool(_VisitWebpageTool):
    """Tool for visiting a webpage."""


@stable
class WikipediaSearchTool(_WikipediaSearchTool):
    """Tool for searching Wikipedia."""


# Re-export the tool functions with their public APIs
@stable
def get_all_default_tools():
    """
    Get all default tools.

    Returns
    -------
        A list of all default tools.

    """
    return _get_all_default_tools()


@stable
def get_default_tool(name: str):
    """
    Get a default tool by name.

    Args:
    ----
        name: The name of the tool to get.

    Returns:
    -------
        The tool with the given name, or None if no such tool exists.

    """
    return _get_default_tool(name)


@stable
def get_registered_tools():
    """
    Get all registered tools.

    Returns
    -------
        A list of all registered tools.

    """
    return _get_registered_tools()


@stable
def is_browser_tools_available():
    """
    Check if browser tools are available.

    Returns
    -------
        True if browser tools are available, False otherwise.

    """
    return _is_browser_tools_available()


@stable
def is_mcp_available():
    """
    Check if MCP tools are available.

    Returns
    -------
        True if MCP tools are available, False otherwise.

    """
    return _is_mcp_available()


@stable
def register_tool(tool):
    """
    Register a tool with the global registry.

    This function registers a tool with the global registry, making it
    available for use by agents. It can register Tool instances, functions,
    or any callable object.

    Args:
    ----
        tool: The tool to register (Tool instance, function, or callable)

    Returns:
    -------
        True if the tool was registered successfully, False otherwise

    Example:
    -------
    ```python
    # Register a Tool instance
    calculator_tool = CalculatorTool()
    register_tool(calculator_tool)

    # Register a function as a tool using the tool decorator
    @tool(name="web_search", description="Search the web for information")
    def search_web(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        # Implementation...
        return results
    ```

    """
    return _register_tool(tool)


# Re-export the tool decorator with its public API
@stable
def tool(*args, **kwargs):
    """
    Decorator for creating tools from functions.

    This decorator provides a convenient way to create tools from functions
    by specifying metadata such as name, description, and parameters.

    Example:
    -------
    ```python
    @tool(name="calculator", description="Performs basic arithmetic operations")
    def calculate(expression: str) -> float:
        \"\"\"
        Calculate the result of a mathematical expression.

    Args:
    ----
            expression (str): The mathematical expression to evaluate

    Returns:
    -------
            float: The result of the calculation
        \"\"\"
        # Implementation...
        return result
    ```

    """
    return _tool(*args, **kwargs)


# Re-export the BrowserManager class with its public API
@beta
class BrowserManager(_BrowserManager):
    """
    Manager for browser instances used by browser tools.

    This class provides functionality for initializing and managing browser
    instances used by browser tools. It handles browser initialization,
    cleanup, and provides access to the browser driver.

    Example:
    -------
    ```python
    # Create a browser manager
    browser_manager = BrowserManager(headless=False)

    # Initialize the browser
    browser_manager.initialize()

    # Use browser tools
    go_to_tool = GoToTool()
    go_to_tool("https://example.com")

    # Close the browser
    browser_manager.close()
    ```

    """


# Re-export the validation functions with their public APIs
@beta
def validate_tool(tool):
    """
    Validate a tool instance.

    This function validates that a tool instance has all the required
    attributes and methods to be used as a tool.

    Args:
    ----
        tool: The tool instance to validate

    Raises:
    ------
        ValueError: If the tool is invalid

    """
    return _validate_tool(tool)


@beta
def validate_tool_attributes(tool_class, check_imports=True):
    """
    Validate that a tool class has all required attributes and follows best practices.

    This function performs comprehensive validation of a tool class, including:
    - Required attributes (name, description, parameters, output_type)
    - Class attribute definitions
    - Method implementations
    - Import usage
    - Parameter validation

    Args:
    ----
        tool_class: The tool class to validate
        check_imports: Whether to check for unauthorized imports

    Raises:
    ------
        ValueError: If the tool class has validation errors

    """
    return _validate_tool_attributes(tool_class, check_imports)


@beta
def validate_tool_parameters(tool, *args, **kwargs):
    """
    Validate that the parameters passed to a tool match its expected inputs.

    This function validates that the parameters passed to a tool match
    its expected inputs, including type checking and required parameter
    validation.

    Args:
    ----
        tool: The tool instance
        *args: Positional arguments
        **kwargs: Keyword arguments

    Raises:
    ------
        ValueError: If the parameters don't match the tool's expected inputs

    """
    return _validate_tool_parameters(tool, *args, **kwargs)
