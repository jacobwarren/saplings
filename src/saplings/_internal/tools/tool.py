from __future__ import annotations

"""
Tool decorator for Saplings.

This module provides a decorator for creating tools from functions.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional

from saplings._internal.tools.base import Tool


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_type: Optional[str] = None,
    **kwargs,
) -> Callable:
    """
    Decorator for creating tools from functions.

    This decorator provides a convenient way to create tools from functions
    by specifying metadata such as name, description, and parameters.

    Args:
    ----
        name: The name of the tool (defaults to the function name)
        description: A description of what the tool does
        parameters: A dictionary of parameter names to parameter metadata
        output_type: The type of the output (e.g., "string", "number", "boolean")
        **kwargs: Additional metadata for the tool

    Returns:
    -------
        Callable: Decorator function

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

    def decorator(func: Callable) -> Callable:
        # Get the function's name and docstring
        func_name = name or func.__name__
        func_doc = description or func.__doc__ or "No description provided"

        # Get the function's signature
        sig = inspect.signature(func)

        # Create parameter metadata if not provided
        if parameters is None:
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self" or param_name == "cls":
                    continue

                param_type = "string"  # Default type
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == str:
                        param_type = "string"
                    elif param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    else:
                        param_type = str(param.annotation)

                params[param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}",
                    "required": param.default == inspect.Parameter.empty,
                }
        else:
            params = parameters

        # Create a Tool instance
        tool_instance = Tool(
            func=func,
            name=func_name,
            description=func_doc,
            parameters=params,
            output_type=output_type,
            **kwargs,
        )

        # Register the tool with the global registry
        from saplings._internal.tools.tool_registry import _TOOL_REGISTRY

        _TOOL_REGISTRY[func_name] = tool_instance

        # Return the original function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add tool metadata to the function
        wrapper.tool = tool_instance

        return wrapper

    return decorator
