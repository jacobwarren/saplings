from __future__ import annotations

"""
Tool decorator for Saplings.

This module provides a decorator to easily convert functions into tools.
"""


import re
import textwrap
from typing import Any, Callable, Union, get_type_hints

from saplings.tools._internal.base import Tool


def _parse_docstring(docstring: str) -> dict[str, Union[str, dict[str, dict[str, str]]]]:
    """
    Parse a docstring to extract description and parameter descriptions.

    Args:
    ----
        docstring: The function docstring

    Returns:
    -------
        A dictionary with 'description' and 'params' keys

    """
    if not docstring:
        return {"description": "", "params": {}}

    # Clean up the docstring
    docstring = textwrap.dedent(docstring).strip()

    # Split into description and parameters
    parts = re.split(r"\n\s*Args:\s*\n", docstring, 1)

    description = parts[0].strip()
    params = {}

    # Parse parameter descriptions if they exist
    if len(parts) > 1:
        param_section = parts[1]
        param_matches = re.finditer(
            r"(\w+)\s*\(([^)]+)\):\s*([^\n]+(?:\n\s+[^\n]+)*)", param_section
        )

        for match in param_matches:
            param_name = match.group(1)
            param_type = match.group(2).strip()
            param_desc = match.group(3).strip()
            params[param_name] = {"type": param_type, "description": param_desc}

    return {"description": description, "params": params}


def _get_type_mapping(type_hint: type) -> str:
    """
    Convert Python type hints to JSON schema types.

    Args:
    ----
        type_hint: The Python type hint

    Returns:
    -------
        The corresponding JSON schema type

    """
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        Any: "any",
        None: "null",
    }

    # Handle Optional types (Union[Type, None])
    if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
        args = type_hint.__args__
        if len(args) == 2 and type(None) in args:
            # This is an Optional[Type]
            for arg in args:
                if arg is not type(None):
                    return _get_type_mapping(arg)

    return type_mapping.get(type_hint, "any")


def tool(name: str | None = None, description: str | None = None) -> Callable:
    """
    Decorator to convert a function into a Tool.

    Args:
    ----
        name: Optional name for the tool (defaults to function name)
        description: Optional description for the tool (defaults to function docstring)

    Returns:
    -------
        A decorator function

    Example:
    -------
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
            # Use a safer alternative to eval
            import ast
            import operator

            def safe_eval(expr):
                # Define supported operations
                operators = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                }

                def _eval(node):
                    if isinstance(node, ast.Num):
                        return node.n
                    elif isinstance(node, ast.BinOp):
                        return operators[type(node.op)](_eval(node.left), _eval(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return operators[type(node.op)](_eval(node.operand))
                    else:
                        raise TypeError(f"Unsupported operation: {node}")

                return _eval(ast.parse(expr, mode='eval').body)

            return safe_eval(expression)

    """

    def decorator(func: Callable) -> Tool:
        # Get function metadata
        func_name = name or func.__name__
        func_doc = func.__doc__ or ""

        # Parse docstring
        doc_info = _parse_docstring(func_doc)
        func_description = description or str(doc_info["description"])

        # Get type hints
        type_hints = get_type_hints(func)
        return_type = type_hints.pop("return", Any)

        # Create inputs dictionary
        inputs = {}
        for param_name, param_type in type_hints.items():
            param_info = {}
            params_dict = doc_info.get("params", {})
            if isinstance(params_dict, dict) and param_name in params_dict:
                param_info = params_dict[param_name]

            inputs[param_name] = {
                "type": _get_type_mapping(param_type),
                "description": param_info.get("description", f"Parameter {param_name}")
                if isinstance(param_info, dict)
                else f"Parameter {param_name}",
                "required": True,  # Default to required
            }

        # Create a Tool subclass
        class FunctionTool(Tool):
            def __init__(self) -> None:
                self.func = func
                super().__init__(func=func, name=func_name, description=func_description)
                self.parameters = inputs
                self.output_type = _get_type_mapping(return_type)

            def forward(self, *args, **kwargs):
                return self.func(*args, **kwargs)

        # Create and return the tool instance
        tool_instance = FunctionTool()

        # Register the tool in the global registry
        from saplings.tools._internal.registry.tool_registry import _TOOL_REGISTRY

        _TOOL_REGISTRY[tool_instance.name] = tool_instance

        return tool_instance

    # Handle case where decorator is used without parentheses
    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator
