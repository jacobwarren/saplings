from __future__ import annotations

"""
Tool registry for Saplings.

This module provides functionality for registering and managing tools that can be used by agents.
"""


import functools
import inspect
import logging
from typing import Callable, Union

logger = logging.getLogger(__name__)


class Tool:
    """
    Base class for tools that can be used by agents.

    A tool is a function that can be called by an agent to perform a specific task.
    Tools have a name, description, and a function that implements the tool's functionality.

    Attributes
    ----------
        name (str): The name of the tool
        description (str): A description of what the tool does
        parameters (dict): Information about the tool's parameters
        output_type (str): The type of the tool's output

    """

    def __init__(
        self,
        func: Callable | None = None,
        name: str | None = None,
        description: str | None = None,
        output_type: str | None = "any",
    ) -> None:
        """
        Initialize a tool.

        Args:
        ----
            func: The function that implements the tool's functionality
            name: The name of the tool (defaults to the function name)
            description: A description of what the tool does
            output_type: The type of the tool's output

        """
        self.is_initialized = False

        if func is not None:
            self.func = func
            self.name = name or func.__name__
            self.description = description or func.__doc__ or "No description provided"
            self.output_type = output_type

            # Get function signature for parameter information
            self.signature = inspect.signature(func)
            self.parameters = {}

            # Extract parameter information
            for param_name, param in self.signature.parameters.items():
                # Skip 'self' parameter for methods
                if param_name == "self" and param.kind == param.POSITIONAL_OR_KEYWORD:
                    continue

                param_info = {
                    "type": "string",  # Default type
                    "description": f"Parameter {param_name}",
                    "required": param.default == param.empty,
                }

                # Try to get better type information
                if param.annotation != param.empty:
                    if param.annotation == str:
                        param_info["type"] = "string"
                    elif param.annotation == int:
                        param_info["type"] = "integer"
                    elif param.annotation == float:
                        param_info["type"] = "number"
                    elif param.annotation == bool:
                        param_info["type"] = "boolean"
                    elif param.annotation in (list, list):
                        param_info["type"] = "array"
                    elif param.annotation in (dict, dict):
                        param_info["type"] = "object"
                    elif hasattr(param.annotation, "__origin__"):
                        # Handle typing generics like Optional, List, etc.
                        origin = param.annotation.__origin__
                        if origin == Union:
                            # Handle Optional (Union[Type, None])
                            args = param.annotation.__args__
                            if type(None) in args and len(args) == 2:
                                # This is Optional[Type]
                                other_type = next(arg for arg in args if arg is not type(None))
                                if other_type == str:
                                    param_info["type"] = "string"
                                elif other_type == int:
                                    param_info["type"] = "integer"
                                elif other_type == float:
                                    param_info["type"] = "number"
                                elif other_type == bool:
                                    param_info["type"] = "boolean"
                                param_info["required"] = False

                self.parameters[param_name] = param_info

        self.is_initialized = True

    def __call__(self, *args, **kwargs):
        """Call the tool's function."""
        if hasattr(self, "func"):
            return self.func(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Implement the tool's functionality.

        This method should be overridden by subclasses that don't provide a function
        in the constructor.

        Args:
        ----
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
        -------
            The result of the tool's execution

        """
        if hasattr(self, "func"):
            return self.func(*args, **kwargs)
        msg = "Subclasses must implement forward() or provide a function in the constructor"
        raise NotImplementedError(msg)

    def setup(self):
        """
        Perform any necessary setup before the tool is used.

        This method can be overridden by subclasses to perform expensive initialization
        that should only happen once.
        """
        self.is_initialized = True

    def to_openai_function(self):
        """
        Convert the tool to an OpenAI function specification.

        Returns
        -------
            Dict: OpenAI function specification

        """
        properties = {}
        required = []

        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info["type"],
                "description": param_info["description"] or f"Parameter {param_name}",
            }

            if param_info["required"]:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }


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
