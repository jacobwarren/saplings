from __future__ import annotations

"""
Function registry module for Saplings.

This module provides a registry for functions that can be called by models.
"""


import inspect
import logging
from typing import Any, Callable, Union, get_type_hints

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """Registry for functions that can be called by models."""

    def __init__(self) -> None:
        """Initialize the function registry."""
        self._functions = {}
        self._function_groups = {}

    def register(
        self,
        func: Callable | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        group: str | None = None,
    ) -> Callable:
        """
        Register a function with the registry.

        Can be used as a decorator:

        ```python
        @function_registry.register(description="Get the weather")
        def get_weather(location: str, unit: str = "celsius") -> dict:
            ...
        ```

        Or called directly:

        ```python
        function_registry.register(get_weather, description="Get the weather")
        ```

        Args:
        ----
            func: The function to register
            name: Name of the function (defaults to function name)
            description: Description of the function
            group: Group to add the function to

        Returns:
        -------
            Callable: The registered function

        """

        def decorator(f: Callable) -> Callable:
            # Get function name
            func_name = name or f.__name__

            # Get function description
            func_description = description or f.__doc__ or f"Function {func_name}"

            # Clean up description
            func_description = inspect.cleandoc(func_description).split("\n")[0]

            # Get function parameters
            func_params = self._get_function_parameters(f)

            # Get required parameters
            required_params = self._get_required_parameters(f)

            # Register the function
            self._functions[func_name] = {
                "function": f,
                "name": func_name,
                "description": func_description,
                "parameters": func_params,
                "required": required_params,
            }

            # Add to group if specified
            if group:
                if group not in self._function_groups:
                    self._function_groups[group] = set()
                self._function_groups[group].add(func_name)

            logger.debug(f"Registered function: {func_name}")
            return f

        # Handle both decorator and direct call
        if func is None:
            return decorator
        return decorator(func)

    def unregister(self, name: str) -> bool:
        """
        Unregister a function from the registry.

        Args:
        ----
            name: Name of the function to unregister

        Returns:
        -------
            bool: True if the function was unregistered, False otherwise

        """
        if name in self._functions:
            # Remove from groups
            for group in self._function_groups.values():
                if name in group:
                    group.remove(name)

            # Remove from functions
            del self._functions[name]
            logger.debug(f"Unregistered function: {name}")
            return True
        return False

    def get_function(self, name: str) -> dict[str, Any] | None:
        """
        Get a function from the registry.

        Args:
        ----
            name: Name of the function

        Returns:
        -------
            Optional[Dict[str, Any]]: The function info or None if not found

        """
        return self._functions.get(name)

    def get_function_definition(self, name: str) -> dict[str, Any] | None:
        """
        Get a function definition for use with LLMs.

        Args:
        ----
            name: Name of the function

        Returns:
        -------
            Optional[Dict[str, Any]]: The function definition or None if not found

        """
        func_info = self._functions.get(name)
        if func_info:
            return {
                "name": func_info["name"],
                "description": func_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": func_info["parameters"],
                    "required": func_info["required"],
                },
            }
        return None

    def get_all_functions(self) -> dict[str, dict[str, Any]]:
        """
        Get all registered functions.

        Returns
        -------
            Dict[str, Dict[str, Any]]: Dictionary of function info

        """
        return self._functions

    def get_group(self, group: str) -> list[str]:
        """
        Get all functions in a group.

        Args:
        ----
            group: Name of the group

        Returns:
        -------
            List[str]: List of function names in the group

        """
        return list(self._function_groups.get(group, set()))

    def get_group_definitions(self, group: str) -> list[dict[str, Any]]:
        """
        Get function definitions for all functions in a group.

        Args:
        ----
            group: Name of the group

        Returns:
        -------
            List[Dict[str, Any]]: List of function definitions

        """
        functions = []
        for name in self.get_group(group):
            func_def = self.get_function_definition(name)
            if func_def:
                functions.append(func_def)
        return functions

    def call_function(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a registered function.

        Args:
        ----
            name: Name of the function to call
            arguments: Arguments to pass to the function

        Returns:
        -------
            Any: The result of the function call

        Raises:
        ------
            ValueError: If the function is not registered
            TypeError: If the arguments are invalid

        """
        func_info = self._functions.get(name)
        if not func_info:
            msg = f"Function not registered: {name}"
            raise ValueError(msg)

        func = func_info["function"]

        # Call the function
        return func(**arguments)

    def _get_function_parameters(self, func: Callable) -> dict[str, Any]:
        """
        Get the parameters of a function.

        Args:
        ----
            func: The function to inspect

        Returns:
        -------
            Dict[str, Any]: Dictionary of parameter info

        """
        params = {}
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        for name, param in sig.parameters.items():
            # Skip self parameter for methods
            if name == "self":
                continue

            # Get parameter type
            param_type = type_hints.get(name, Any)

            # Get default value
            has_default = param.default is not param.empty
            default_value = param.default if has_default else None

            # Get parameter description from docstring
            param_desc = self._get_parameter_description(func, name)

            # Create parameter info
            param_info = self._create_parameter_info(name, param_type, default_value, param_desc)
            params[name] = param_info

        return params

    def _get_required_parameters(self, func: Callable) -> list[str]:
        """
        Get the required parameters of a function.

        Args:
        ----
            func: The function to inspect

        Returns:
        -------
            List[str]: List of required parameter names

        """
        required = []
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            # Skip self parameter for methods
            if name == "self":
                continue

            # Check if parameter is required
            if param.default is param.empty and param.kind not in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                required.append(name)

        return required

    def _get_parameter_description(self, func: Callable, param_name: str) -> str:
        """
        Get the description of a parameter from the function's docstring.

        Args:
        ----
            func: The function to inspect
            param_name: Name of the parameter

        Returns:
        -------
            str: Description of the parameter

        """
        if not func.__doc__:
            return f"Parameter {param_name}"

        docstring = inspect.cleandoc(func.__doc__)
        lines = docstring.split("\n")

        # Look for parameter in docstring
        param_pattern = f"{param_name}:"
        for i, line in enumerate(lines):
            if param_pattern in line:
                # Extract description
                desc = line.split(":", 1)[1].strip()

                # Check for multi-line description
                j = i + 1
                while j < len(lines) and lines[j].startswith(" "):
                    desc += " " + lines[j].strip()
                    j += 1

                return desc

        return f"Parameter {param_name}"

    def _create_parameter_info(
        self, name: str, param_type: type, default_value: Any, description: str
    ) -> dict[str, Any]:
        """
        Create parameter info for a function parameter.

        Args:
        ----
            name: Name of the parameter
            param_type: Type of the parameter
            default_value: Default value of the parameter
            description: Description of the parameter

        Returns:
        -------
            Dict[str, Any]: Parameter info

        """
        # Handle basic types
        if param_type == str:
            param_info = {"type": "string", "description": description}
        elif param_type == int:
            param_info = {"type": "integer", "description": description}
        elif param_type == float:
            param_info = {"type": "number", "description": description}
        elif param_type == bool:
            param_info = {"type": "boolean", "description": description}
        elif param_type == list:
            param_info = {"type": "array", "description": description}
        elif param_type == dict:
            param_info = {"type": "object", "description": description}
        else:
            # Default to string for complex types
            param_info = {"type": "string", "description": description}

        # Add default value if available
        if default_value is not None and default_value is not inspect.Parameter.empty:
            param_info["default"] = default_value

        # Add enum if the type is a string literal with limited values
        if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
            # Handle Union types like Literal
            if hasattr(param_type, "__args__"):
                args = param_type.__args__
                if all(isinstance(arg, str) for arg in args):
                    param_info = {
                        **param_info,
                        "type": "string",
                        "enum": [str(arg) for arg in args],
                    }

        return param_info


def register_function(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    group: str | None = None,
) -> Callable:
    """
    Register a function with the registry.

    This is a convenience function that calls FunctionRegistry.register.
    It gets the registry instance from the DI container.

    Args:
    ----
        func: The function to register
        name: Name of the function (defaults to function name)
        description: Description of the function
        group: Group to add the function to

    Returns:
    -------
        Callable: The registered function

    """
    from saplings.di import container

    registry = container.resolve(FunctionRegistry)
    return registry.register(
        func,
        name=name,
        description=description,
        group=group,
    )


def get_function_registry():
    """
    Get the function registry instance.

    This function is maintained for backward compatibility.
    New code should use constructor injection via the DI container.

    Returns
    -------
        FunctionRegistry: Function registry instance from the DI container

    """
    from saplings.di import container

    return container.resolve(FunctionRegistry)


# Create a global function registry instance for backward compatibility
function_registry = FunctionRegistry()
