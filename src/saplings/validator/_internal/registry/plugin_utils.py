from __future__ import annotations

"""
Plugin utilities for validator registry.

This module provides utilities for working with plugins to avoid circular imports.
"""

from enum import Enum
from typing import Any, Dict, Protocol, Type, runtime_checkable


class PluginTypeEnum(str, Enum):
    """Types of plugins supported by Saplings."""

    MODEL_ADAPTER = "model_adapter"
    MEMORY_STORE = "memory_store"
    VALIDATOR = "validator"
    INDEXER = "indexer"
    TOOL = "tool"


@runtime_checkable
class PluginProtocol(Protocol):
    """Protocol for plugins to avoid circular imports."""

    @property
    def name(self) -> str:
        """Name of the plugin."""
        ...

    @property
    def version(self) -> str:
        """Version of the plugin."""
        ...

    @property
    def description(self) -> str:
        """Description of the plugin."""
        ...

    @property
    def plugin_type(self) -> PluginTypeEnum:
        """Type of the plugin."""
        ...


def get_plugins_by_type_lazy(plugin_type: PluginTypeEnum) -> Dict[str, Type[Any]]:
    """
    Get all plugins of a specific type using lazy imports to avoid circular dependencies.

    Args:
    ----
        plugin_type: Type of plugins to get

    Returns:
    -------
        Dict[str, Type[Any]]: Dictionary of plugin name to plugin class

    """

    # Use lazy imports to avoid circular dependencies
    def _get_plugins():
        from saplings.core._internal.plugin import PluginType, get_plugins_by_type

        # Map our enum to the core enum
        core_plugin_type = getattr(PluginType, plugin_type.value.upper())

        # Get the plugins
        return get_plugins_by_type(core_plugin_type)

    # Call the function to get the plugins
    return _get_plugins()
