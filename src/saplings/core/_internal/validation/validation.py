from __future__ import annotations

"""
Parameter validation utilities for Saplings.

This module provides utilities for validating parameters and ensuring
that they meet the required constraints.
"""

import inspect
from typing import Any, Callable, Optional, TypeVar, get_type_hints

from saplings.core._internal.exceptions import ConfigurationError

T = TypeVar("T")


def validate_required(value: Any, name: str) -> None:
    """
    Validate that a required parameter is not None.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages

    Raises:
    ------
        ConfigurationError: If the parameter is None

    """
    if value is None:
        raise ConfigurationError(
            f"Required parameter '{name}' cannot be None",
            config_key=name,
            config_value=value,
        )


def validate_not_empty(value: Any, name: str) -> None:
    """
    Validate that a parameter is not empty.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages

    Raises:
    ------
        ConfigurationError: If the parameter is empty

    """
    if value is None:
        raise ConfigurationError(
            f"Parameter '{name}' cannot be None",
            config_key=name,
            config_value=value,
        )

    if isinstance(value, str) and not value.strip():
        raise ConfigurationError(
            f"Parameter '{name}' cannot be empty",
            config_key=name,
            config_value=value,
        )

    if hasattr(value, "__len__") and len(value) == 0:
        raise ConfigurationError(
            f"Parameter '{name}' cannot be empty",
            config_key=name,
            config_value=value,
        )


def validate_positive(value: float, name: str) -> None:
    """
    Validate that a parameter is positive.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages

    Raises:
    ------
        ConfigurationError: If the parameter is not positive

    """
    if value <= 0:
        raise ConfigurationError(
            f"Parameter '{name}' must be positive",
            config_key=name,
            config_value=value,
        )


def validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a parameter is non-negative.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages

    Raises:
    ------
        ConfigurationError: If the parameter is negative

    """
    if value < 0:
        raise ConfigurationError(
            f"Parameter '{name}' cannot be negative",
            config_key=name,
            config_value=value,
        )


def validate_in_range(value: float, name: str, min_value: float, max_value: float) -> None:
    """
    Validate that a parameter is within a specified range.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages
        min_value: The minimum allowed value (inclusive)
        max_value: The maximum allowed value (inclusive)

    Raises:
    ------
        ConfigurationError: If the parameter is outside the allowed range

    """
    if value < min_value or value > max_value:
        raise ConfigurationError(
            f"Parameter '{name}' must be between {min_value} and {max_value}",
            config_key=name,
            config_value=value,
        )


def validate_one_of(value: Any, name: str, allowed_values: list[Any]) -> None:
    """
    Validate that a parameter is one of the allowed values.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages
        allowed_values: List of allowed values

    Raises:
    ------
        ConfigurationError: If the parameter is not one of the allowed values

    """
    if value not in allowed_values:
        raise ConfigurationError(
            f"Parameter '{name}' must be one of {allowed_values}",
            config_key=name,
            config_value=value,
        )


def validate_type(value: Any, name: str, expected_type: type) -> None:
    """
    Validate that a parameter is of the expected type.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages
        expected_type: The expected type

    Raises:
    ------
        ConfigurationError: If the parameter is not of the expected type

    """
    if value is not None and not isinstance(value, expected_type):
        raise ConfigurationError(
            f"Parameter '{name}' must be of type {expected_type.__name__}",
            config_key=name,
            config_value=value,
        )


def validate_callable(value: Any, name: str) -> None:
    """
    Validate that a parameter is callable.

    Args:
    ----
        value: The parameter value to validate
        name: The name of the parameter for error messages

    Raises:
    ------
        ConfigurationError: If the parameter is not callable

    """
    if value is not None and not callable(value):
        raise ConfigurationError(
            f"Parameter '{name}' must be callable",
            config_key=name,
            config_value=value,
        )


def optional_param(value: Optional[T], default: T) -> T:
    """
    Handle an optional parameter with a default value.

    Args:
    ----
        value: The parameter value or None
        default: The default value to use if value is None

    Returns:
    -------
        The parameter value or the default value

    """
    return value if value is not None else default


def validate_parameters(func: Callable, **kwargs) -> None:
    """
    Validate parameters against a function's type hints.

    Args:
    ----
        func: The function to validate parameters against
        **kwargs: The parameters to validate

    Raises:
    ------
        ConfigurationError: If a parameter doesn't match its type hint

    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    for name, param in sig.parameters.items():
        if name in kwargs and name in type_hints:
            value = kwargs[name]
            hint = type_hints[name]

            # Skip validation for None values in Optional parameters
            if value is None and getattr(hint, "__origin__", None) is Optional:
                continue

            # Get the actual type for Optional parameters
            if getattr(hint, "__origin__", None) is Optional:
                hint = hint.__args__[0]

            # Validate the parameter type
            if not isinstance(value, hint):
                raise ConfigurationError(
                    f"Parameter '{name}' must be of type {hint.__name__}",
                    config_key=name,
                    config_value=value,
                )
