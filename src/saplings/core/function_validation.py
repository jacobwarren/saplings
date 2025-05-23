from __future__ import annotations

"""
Function validation module for Saplings.

This module provides utilities for validating function calls.
"""


import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from saplings.core.function_registry import function_registry

logger = logging.getLogger(__name__)


class FunctionValidator:
    """Utility for validating function calls."""

    def __init__(self) -> None:
        """Initialize the function validator."""
        self._validation_models = {}

    def validate_function_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a function call.

        Args:
        ----
            name: Name of the function to validate
            arguments: Arguments to validate

        Returns:
        -------
            Dict[str, Any]: Validated and possibly converted arguments

        Raises:
        ------
            ValueError: If the function is not registered
            ValidationError: If the arguments are invalid

        """
        # Get the function
        func_info = function_registry.get_function(name)
        if not func_info:
            msg = f"Function not registered: {name}"
            raise ValueError(msg)

        # Parse arguments if they're a string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON in arguments: {e}"
                raise ValueError(msg)

        # Get or create the validation model
        validation_model = self._get_validation_model(name, func_info)

        # Validate the arguments
        try:
            validated = validation_model(**arguments)
            return validated.model_dump()
        except ValidationError as e:
            logger.exception(f"Validation error for function {name}: {e}")
            raise

    def _get_validation_model(self, name: str, func_info: dict[str, Any]) -> type[BaseModel]:
        """
        Get or create a validation model for a function.

        Args:
        ----
            name: Name of the function
            func_info: Function info from the registry

        Returns:
        -------
            Type[BaseModel]: Validation model

        """
        if name in self._validation_models:
            return self._validation_models[name]

        # Get function and parameters
        func_info["function"]
        parameters = func_info["parameters"]
        required = set(func_info["required"])

        # Create field definitions
        fields = {}
        for param_name, param_info in parameters.items():
            # Skip self parameter for methods
            if param_name == "self":
                continue

            # Get parameter type
            param_type = self._get_parameter_type(param_info)

            # Check if parameter is required
            is_required = param_name in required

            # Get default value
            default = param_info.get("default", ... if is_required else None)

            # Create field
            fields[param_name] = (param_type, default)

        # Create the model
        model_name = f"{name.title()}Args"
        validation_model = create_model(model_name, **fields)

        # Cache the model
        self._validation_models[name] = validation_model

        return validation_model

    def _get_parameter_type(self, param_info: dict[str, Any]) -> type:
        """
        Get the type for a parameter.

        Args:
        ----
            param_info: Parameter info

        Returns:
        -------
            Type: Parameter type

        """
        param_type = param_info.get("type", "string")

        # Map parameter type to Python type
        if param_type == "string":
            return str
        if param_type == "integer":
            return int
        if param_type == "number":
            return float
        if param_type == "boolean":
            return bool
        if param_type == "array":
            return list[Any]
        if param_type == "object":
            return dict[str, Any]
        return object


# Create a singleton instance
function_validator = FunctionValidator()


def validate_function_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a function call.

    This is a convenience function that uses the FunctionValidator.

    Args:
    ----
        name: Name of the function to validate
        arguments: Arguments to validate

    Returns:
    -------
        Dict[str, Any]: Validated and possibly converted arguments

    Raises:
    ------
        ValueError: If the function is not registered
        ValidationError: If the arguments are invalid

    """
    return function_validator.validate_function_call(name, arguments)
