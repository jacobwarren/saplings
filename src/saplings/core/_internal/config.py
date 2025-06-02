from __future__ import annotations

"""
Core configuration utilities for Saplings.

This module provides minimal configuration helpers without any external I/O dependencies.
Configuration loading from external sources (env vars, files) is handled in higher-level modules.
"""


from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class ConfigValue(Generic[T]):
    """
    A configuration value with optional default.

    This class provides a clean way to define and access configuration values
    with defaults and validation, without external dependencies.
    """

    def __init__(
        self,
        default: T | None = None,
        validator: Callable[[T], bool] | None = None,
        description: str = "",
    ) -> None:
        """
        Initialize a configuration value.

        Args:
        ----
            default: Default value if none is provided
            validator: Optional function to validate the value
            description: Human-readable description of the config value

        """
        self._value: T | None = default
        self._default: T | None = default
        self._validator = validator
        self._description = description

    def get(self):
        """
        Get the configuration value.

        Returns
        -------
            The configuration value or default

        """
        return self._value

    def set(self, value: T) -> None:
        """
        Set the configuration value.

        Args:
        ----
            value: New value to set

        Raises:
        ------
            ValueError: If validation fails

        """
        if self._validator is not None and value is not None and not self._validator(value):
            msg = f"Invalid configuration value: {value}"
            raise ValueError(msg)

        self._value = value

    def reset(self):
        """Reset to default value."""
        self._value = self._default

    @property
    def description(self) -> str:
        """Get the human-readable description."""
        return self._description


class Config:
    """
    Simple configuration container.

    This class provides a container for configuration values without external
    dependencies. Higher-level modules can extend this with environment loading,
    file loading, etc.
    """

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        """
        Initialize a configuration container.

        Args:
        ----
            values: Optional initial values

        """
        self._values: dict[str, Any] = values or {}

    def get(self, key: str, default: T | None = None) -> T | None:
        """
        Get a configuration value.

        Args:
        ----
            key: Configuration key
            default: Default value if the key doesn't exist

        Returns:
        -------
            The configuration value or default

        """
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
        ----
            key: Configuration key
            value: Configuration value

        """
        self._values[key] = value

    def update(self, values: dict[str, Any]) -> None:
        """
        Update multiple configuration values.

        Args:
        ----
            values: Dictionary of values to update

        """
        self._values.update(values)

    def as_dict(self):
        """
        Get all configuration values as a dictionary.

        Returns
        -------
            Dictionary of all configuration values

        """
        return self._values.copy()


def create_validator_chain(*validators: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """
    Create a chain of validators.

    Args:
    ----
        *validators: Validator functions

    Returns:
    -------
        A function that returns True if all validators return True

    """

    def validate(value: Any) -> bool:
        return all(validator(value) for validator in validators)

    return validate


def create_type_validator(*types: type) -> Callable[[Any], bool]:
    """
    Create a validator that checks the type of a value.

    Args:
    ----
        *types: Valid types

    Returns:
    -------
        A function that returns True if the value is one of the specified types

    """

    def validate(value: Any) -> bool:
        return isinstance(value, types) if value is not None else True

    return validate


def create_range_validator(
    min_value: float | None = None, max_value: float | None = None
) -> Callable[[int | float], bool]:
    """
    Create a validator that checks if a numeric value is within a range.

    Args:
    ----
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)

    Returns:
    -------
        A function that returns True if the value is within the range

    """

    def validate(value: float) -> bool:
        if value is None:
            return True

        if min_value is not None and value < min_value:
            return False

        return not (max_value is not None and value > max_value)

    return validate


def create_enum_validator(valid_values: list[Any]) -> Callable[[Any], bool]:
    """
    Create a validator that checks if a value is in a list of valid values.

    Args:
    ----
        valid_values: List of valid values

    Returns:
    -------
        A function that returns True if the value is in the list

    """

    def validate(value: Any) -> bool:
        return value in valid_values if value is not None else True

    return validate
