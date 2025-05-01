"""
Plugin system for Saplings.

This module provides the plugin discovery and loading mechanism for Saplings.
"""

import importlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

from importlib_metadata import entry_points

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Types of plugins supported by Saplings."""

    MODEL_ADAPTER = "model_adapter"
    MEMORY_STORE = "memory_store"
    VALIDATOR = "validator"
    INDEXER = "indexer"
    TOOL = "tool"


class Plugin(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the plugin."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the plugin."""
        pass

    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        pass


T = TypeVar("T", bound=Plugin)


class PluginRegistry:
    """Registry for plugins."""

    _instance: Optional["PluginRegistry"] = None
    _plugins: Dict[PluginType, Dict[str, Type[Plugin]]] = {}

    def __new__(cls) -> "PluginRegistry":
        """Create a singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = super(PluginRegistry, cls).__new__(cls)
            # Initialize plugin dictionaries
            cls._instance._plugins = {}
            for plugin_type in PluginType:
                cls._instance._plugins[plugin_type] = {}
        return cls._instance

    def register_plugin(self, plugin_class: Type[Plugin]) -> None:
        """
        Register a plugin class.

        Args:
            plugin_class: The plugin class to register
        """
        # Create a temporary instance to access property values
        temp_instance = plugin_class()
        plugin_type = temp_instance.plugin_type
        plugin_name = temp_instance.name

        if plugin_name in self._plugins[plugin_type]:
            logger.warning(
                f"Plugin {plugin_name} of type {plugin_type} is already registered. Overwriting."
            )

        self._plugins[plugin_type][plugin_name] = plugin_class
        logger.info(f"Registered plugin {plugin_name} of type {plugin_type}")

    def get_plugin(self, plugin_type: PluginType, plugin_name: str) -> Optional[Type[Plugin]]:
        """
        Get a plugin by type and name.

        Args:
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin

        Returns:
            Optional[Type[Plugin]]: The plugin class if found, None otherwise
        """
        return self._plugins.get(plugin_type, {}).get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, Type[Plugin]]:
        """
        Get all plugins of a specific type.

        Args:
            plugin_type: Type of plugins to get

        Returns:
            Dict[str, Type[Plugin]]: Dictionary of plugin name to plugin class
        """
        return self._plugins.get(plugin_type, {})

    def get_plugin_types(self) -> List[PluginType]:
        """
        Get all plugin types.

        Returns:
            List[PluginType]: List of plugin types
        """
        return list(PluginType)

    def clear(self) -> None:
        """Clear all registered plugins."""
        for plugin_type in PluginType:
            self._plugins[plugin_type] = {}
        logger.info("Cleared all registered plugins")


def discover_plugins() -> None:
    """
    Discover and register plugins from entry points.

    This function looks for entry points in the following groups:
    - saplings.model_adapters
    - saplings.memory_stores
    - saplings.validators
    - saplings.indexers
    - saplings.tools
    """
    registry = PluginRegistry()

    # Map entry point groups to plugin types
    entry_point_groups = {
        "saplings.model_adapters": PluginType.MODEL_ADAPTER,
        "saplings.memory_stores": PluginType.MEMORY_STORE,
        "saplings.validators": PluginType.VALIDATOR,
        "saplings.indexers": PluginType.INDEXER,
        "saplings.tools": PluginType.TOOL,
    }

    # Discover plugins from entry points
    for group_name, plugin_type in entry_point_groups.items():
        for entry_point in entry_points(group=group_name):
            try:
                plugin_class = entry_point.load()
                if not issubclass(plugin_class, Plugin):
                    logger.warning(
                        f"Entry point {entry_point.name} in group {group_name} "
                        f"does not provide a Plugin subclass"
                    )
                    continue

                # Verify that the plugin type matches the entry point group
                if plugin_class.plugin_type != plugin_type:
                    logger.warning(
                        f"Plugin {plugin_class.name} has type {plugin_class.plugin_type} "
                        f"but was registered in group {group_name} for type {plugin_type}"
                    )
                    continue

                registry.register_plugin(plugin_class)
            except Exception as e:
                logger.error(
                    f"Error loading plugin {entry_point.name} from group {group_name}: {e}"
                )

    logger.info("Plugin discovery completed")


def get_plugin_registry() -> PluginRegistry:
    """
    Get the plugin registry.

    Returns:
        PluginRegistry: The plugin registry
    """
    return PluginRegistry()


def get_plugin(plugin_type: PluginType, plugin_name: str) -> Optional[Type[Plugin]]:
    """
    Get a plugin by type and name.

    Args:
        plugin_type: Type of the plugin
        plugin_name: Name of the plugin

    Returns:
        Optional[Type[Plugin]]: The plugin class if found, None otherwise
    """
    return get_plugin_registry().get_plugin(plugin_type, plugin_name)


def get_plugins_by_type(plugin_type: PluginType) -> Dict[str, Type[Plugin]]:
    """
    Get all plugins of a specific type.

    Args:
        plugin_type: Type of plugins to get

    Returns:
        Dict[str, Type[Plugin]]: Dictionary of plugin name to plugin class
    """
    return get_plugin_registry().get_plugins_by_type(plugin_type)


def register_plugin(plugin_class: Type[Plugin]) -> None:
    """
    Register a plugin class.

    Args:
        plugin_class: The plugin class to register
    """
    get_plugin_registry().register_plugin(plugin_class)


# Type-specific plugin base classes


class ModelAdapterPlugin(Plugin):
    """Base class for model adapter plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MODEL_ADAPTER


class MemoryStorePlugin(Plugin):
    """Base class for memory store plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.MEMORY_STORE


class ValidatorPlugin(Plugin):
    """Base class for validator plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.VALIDATOR


class IndexerPlugin(Plugin):
    """Base class for indexer plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.INDEXER


class ToolPlugin(Plugin):
    """Base class for tool plugins."""

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        return PluginType.TOOL
