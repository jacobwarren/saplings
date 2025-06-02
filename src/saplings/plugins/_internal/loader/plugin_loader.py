from __future__ import annotations

"""
Plugin loader module for Saplings.

This module provides functionality for loading and registering plugins.
It includes utilities for registering built-in plugins and handling plugin
registration errors.
"""

from typing import Optional

from saplings.api.registry import PluginRegistry, RegistrationMode, register_plugin
from saplings.api.validator import ExecutionValidator
from saplings.plugins.indexers.code_indexer import CodeIndexer
from saplings.plugins.memory_stores.secure_memory_store import SecureMemoryStore
from saplings.plugins.validators.code_validator import CodeValidator
from saplings.plugins.validators.factual_validator import FactualValidator


class PluginLoadError(Exception):
    """Exception raised when a plugin fails to load."""


def register_all_plugins(
    registry: Optional[PluginRegistry] = None, mode: RegistrationMode = RegistrationMode.SKIP
) -> tuple[int, int]:
    """
    Register all built-in plugins.

    This function registers all built-in plugins with the plugin registry.
    It uses a try-except block for each plugin to ensure that failure to
    register one plugin doesn't prevent others from being registered.

    Args:
    ----
        registry: Optional registry to use for registration
        mode: Registration mode to use (default: SKIP to avoid warnings)

    Returns:
    -------
        A tuple of (success_count, failure_count)

    Raises:
    ------
        PluginLoadError: If a plugin fails to load and strict mode is enabled

    """
    import logging

    logger = logging.getLogger(__name__)

    # Track registration success
    success_count = 0
    failure_count = 0

    # Helper function to register a plugin with error handling
    def register_with_error_handling(plugin_class, plugin_type):
        nonlocal success_count, failure_count
        try:
            register_plugin(plugin_class, registry=registry, mode=mode)
            success_count += 1
            return True
        except Exception as e:
            logger.error(f"Failed to register {plugin_type} plugin {plugin_class.__name__}: {e}")
            failure_count += 1
            return False

    # Register memory store plugins
    register_with_error_handling(SecureMemoryStore, "memory store")

    # Register validator plugins
    register_with_error_handling(CodeValidator, "validator")
    register_with_error_handling(FactualValidator, "validator")
    register_with_error_handling(ExecutionValidator, "validator")

    # Register indexer plugins
    register_with_error_handling(CodeIndexer, "indexer")

    # Log registration results
    logger.info(f"Plugin registration complete: {success_count} succeeded, {failure_count} failed")

    # Return the counts
    return success_count, failure_count


class PluginLoader:
    """
    Plugin loader for Saplings.

    This class provides functionality for loading and registering plugins.
    """

    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the plugin loader.

        Args:
        ----
            registry: Optional registry to use for registration

        """
        self.registry = registry

    def register_all(self, mode: RegistrationMode = RegistrationMode.SKIP) -> tuple[int, int]:
        """
        Register all built-in plugins.

        Args:
        ----
            mode: Registration mode to use

        Returns:
        -------
            A tuple of (success_count, failure_count)

        """
        return register_all_plugins(self.registry, mode)


if __name__ == "__main__":
    success, failure = register_all_plugins()
    print(f"Registration complete: {success} succeeded, {failure} failed")
