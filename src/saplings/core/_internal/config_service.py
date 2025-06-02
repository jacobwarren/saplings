from __future__ import annotations

"""
Configuration service for Saplings.

This module provides a central configuration service that abstracts environment
variable access and provides a more reproducible approach to configuration.
"""


import logging
import os
from typing import Optional, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigurationService:
    """
    Configuration service for Saplings.

    This service provides a central place to manage configuration settings,
    abstracting away direct environment variable access and improving
    reproducibility. It supports:

    1. Default values
    2. Type conversion
    3. Validation
    4. Overrides for testing
    5. Consistent access patterns
    """

    def __init__(self) -> None:
        """Initialize the configuration service."""
        self._config_cache: dict[str, object] = {}
        self._override_values: dict[str, object] = {}

    def get_value(self, key: str, default: T | None = None, value_type: type[T] = str) -> T | None:
        """
        Get a configuration value.

        Args:
        ----
            key: The configuration key
            default: Default value if not found
            value_type: Type to convert value to

        Returns:
        -------
            The configuration value or default

        """
        # Check if the value is in the override map (for testing)
        if key in self._override_values:
            return cast("Optional[T]", self._override_values.get(key))

        # Check if the value is already in the cache
        if key in self._config_cache:
            return cast("Optional[T]", self._config_cache.get(key))

        # Look for environment variable
        env_value = os.environ.get(key)

        # If not found, return default
        if env_value is None:
            return default

        # Convert value to the requested type
        try:
            if value_type == bool:
                # Handle boolean conversion
                typed_value = env_value.lower() in ("true", "1", "yes", "y")
            elif value_type == int:
                typed_value = int(env_value)
            elif value_type == float:
                typed_value = float(env_value)
            elif value_type == list:
                # Split comma-separated values
                typed_value = [item.strip() for item in env_value.split(",")]
            elif value_type == dict:
                # Parse JSON string
                import json

                typed_value = json.loads(env_value)
            else:
                # Default to string
                typed_value = env_value

            # Cache the value
            self._config_cache[key] = typed_value
            return cast("T", typed_value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting config value for {key}: {e}")
            return default

    def set_override(self, key: str, value: object) -> None:
        """
        Set an override value for testing.

        Args:
        ----
            key: Configuration key
            value: Override value

        """
        self._override_values[key] = value

    def clear_override(self, key: str) -> None:
        """
        Clear an override value.

        Args:
        ----
            key: Configuration key to clear

        """
        if key in self._override_values:
            del self._override_values[key]

    def clear_all_overrides(self):
        """Clear all override values."""
        self._override_values.clear()


# Global singleton instance
config_service = ConfigurationService()
