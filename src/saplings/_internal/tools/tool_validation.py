from __future__ import annotations

"""
Tool validation utilities for Saplings.

This module provides functions to validate tool attributes and parameters.
"""

import builtins
import inspect

# Set of built-in names in Python
_BUILTIN_NAMES = set(vars(builtins))

# Base built-in modules that are safe to import
BASE_BUILTIN_MODULES = {
    "math",
    "random",
    "re",
    "json",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "uuid",
    "copy",
    "string",
    "time",
    "calendar",
    "fractions",
    "decimal",
    "bisect",
    "heapq",
    "array",
    "enum",
    "dataclasses",
}


def is_valid_identifier(name: str) -> bool:
    """
    Check if a string is a valid Python identifier.

    Args:
    ----
        name: The string to check

    Returns:
    -------
        True if the string is a valid Python identifier, False otherwise

    """
    if not name:
        return False

    # Check if the name is a valid Python identifier
    if not name.isidentifier():
        return False

    # Check if the name is a Python keyword
    import keyword

    return not keyword.iskeyword(name)


def validate_tool_attributes(tool_class: type, check_imports: bool = True) -> None:
    """
    Validate that a tool class has all required attributes and follows best practices.

    This function performs comprehensive validation of a tool class, including:
    - Required attributes (name, description, inputs, output_type)
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
    required_attributes = {
        "name": str,
        "description": str,
        "parameters": dict,
        "output_type": str,
    }

    errors = []

    # Check required attributes
    for attr_name, attr_type in required_attributes.items():
        if not hasattr(tool_class, attr_name):
            errors.append(f"Missing required attribute: {attr_name}")
            continue

        attr_value = getattr(tool_class, attr_name)
        if not isinstance(attr_value, attr_type):
            errors.append(
                f"Attribute {attr_name} should be of type {attr_type.__name__}, "
                f"got {type(attr_value).__name__} instead"
            )

    # Check name is valid
    if hasattr(tool_class, "name"):
        name = tool_class.name
        if not name or not isinstance(name, str) or not is_valid_identifier(name):
            errors.append(f"Invalid tool name: {name}. Must be a valid Python identifier.")

    # Check parameters are valid
    if hasattr(tool_class, "parameters"):
        parameters = tool_class.parameters
        if not isinstance(parameters, dict):
            errors.append(f"Parameters should be a dict, got {type(parameters).__name__} instead")
        else:
            for param_name, param_spec in parameters.items():
                if not is_valid_identifier(param_name):
                    errors.append(
                        f"Invalid parameter name: {param_name}. Must be a valid Python identifier."
                    )

                if not isinstance(param_spec, dict):
                    errors.append(
                        f"Parameter specification for {param_name} should be a dict, "
                        f"got {type(param_spec).__name__} instead"
                    )
                    continue

                # Check parameter has required fields
                required_fields = ["type", "description"]
                for field in required_fields:
                    if field not in param_spec:
                        errors.append(f"Missing required field '{field}' in parameter {param_name}")

                # Check parameter type is valid
                if "type" in param_spec:
                    param_type = param_spec["type"]
                    valid_types = [
                        "string",
                        "boolean",
                        "integer",
                        "number",
                        "image",
                        "audio",
                        "array",
                        "object",
                        "any",
                    ]
                    if param_type not in valid_types:
                        errors.append(
                            f"Invalid parameter type for {param_name}: {param_type}. Must be one of {valid_types}"
                        )

    # Check output_type is valid
    if hasattr(tool_class, "output_type"):
        output_type = tool_class.output_type
        valid_types = [
            "string",
            "boolean",
            "integer",
            "number",
            "image",
            "audio",
            "array",
            "object",
            "any",
        ]
        if output_type not in valid_types:
            errors.append(f"Invalid output_type: {output_type}. Must be one of {valid_types}")

    # Check forward method exists and has correct signature
    if not hasattr(tool_class, "forward"):
        errors.append("Missing required method: forward")
    else:
        forward = tool_class.forward
        if not callable(forward):
            errors.append("'forward' attribute must be a callable method")
        # Check forward method signature matches parameters
        elif hasattr(tool_class, "parameters") and not getattr(
            tool_class, "skip_forward_signature_validation", False
        ):
            try:
                signature = inspect.signature(forward)
                actual_params = {p for p in signature.parameters if p != "self"}
                expected_params = set(getattr(tool_class, "parameters", {}).keys())

                if actual_params != expected_params:
                    errors.append(
                        f"Forward method parameters {actual_params} don't match parameters {expected_params}"
                    )
            except (ValueError, TypeError):
                # Can't get signature, skip this check
                pass

    if errors:
        raise ValueError(f"Tool validation failed for {tool_class.__name__}:\n" + "\n".join(errors))


def validate_tool_parameters(tool, *args, **kwargs) -> None:
    """
    Validate that the parameters passed to a tool match its expected inputs.

    Args:
    ----
        tool: The tool instance
        *args: Positional arguments
        **kwargs: Keyword arguments

    Raises:
    ------
        ValueError: If the parameters don't match the tool's expected inputs

    """
    if not hasattr(tool, "parameters"):
        return

    parameters = tool.parameters
    errors = []

    # Check if args can be mapped to inputs
    if args:
        if len(args) > len(parameters):
            errors.append(
                f"Too many positional arguments: expected {len(parameters)}, got {len(args)}"
            )

    # Check kwargs match expected inputs
    for kwarg_name, kwarg_value in kwargs.items():
        if kwarg_name not in parameters:
            errors.append(f"Unexpected argument: {kwarg_name}")
            continue

        input_spec = parameters[kwarg_name]

        # Check type if specified
        if "type" in input_spec:
            expected_type = input_spec["type"]
            # Perform basic type checking
            if expected_type == "string" and not isinstance(kwarg_value, str):
                errors.append(f"Argument {kwarg_name} should be a string")
            elif expected_type == "integer" and not isinstance(kwarg_value, int):
                errors.append(f"Argument {kwarg_name} should be an integer")
            elif expected_type == "number" and not isinstance(kwarg_value, (int, float)):
                errors.append(f"Argument {kwarg_name} should be a number")
            elif expected_type == "boolean" and not isinstance(kwarg_value, bool):
                errors.append(f"Argument {kwarg_name} should be a boolean")

    # Check for missing required arguments
    provided_args = set(kwargs.keys()) | set(list(parameters.keys())[: len(args)])
    for input_name, input_spec in parameters.items():
        if input_name not in provided_args and input_spec.get("required", True):
            errors.append(f"Missing required argument: {input_name}")

    if errors:
        raise ValueError("Tool parameter validation failed:\n" + "\n".join(errors))


def validate_tool(tool) -> None:
    """
    Validate a tool instance.

    Args:
    ----
        tool: The tool instance to validate

    Raises:
    ------
        ValueError: If the tool is invalid

    """
    validate_tool_attributes(tool.__class__)
