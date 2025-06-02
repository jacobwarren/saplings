from __future__ import annotations

"""
Plugin registry for Saplings.

This module provides the plugin registry for Saplings.
"""

import logging
import threading
from enum import Enum
from typing import Dict, Optional, Type

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Types of plugins supported by Saplings."""

    MODEL_ADAPTER = "model_adapter"
    MEMORY_STORE = "memory_store"
    VALIDATOR = "validator"
    INDEXER = "indexer"
    TOOL = "tool"


class RegistrationMode(str, Enum):
    """Plugin registration modes."""

    STRICT = "strict"  # Raise error if plugin already exists
    WARN = "warn"  # Log warning if plugin already exists (current behavior)
    SILENT = "silent"  # Silently overwrite existing plugin
    SKIP = "skip"  # Skip registration if plugin already exists


class Plugin:
    """Base class for all plugins."""

    @property
    def name(self) -> str:
        """Name of the plugin."""
        raise NotImplementedError

    @property
    def version(self) -> str:
        """Version of the plugin."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Description of the plugin."""
        raise NotImplementedError

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        raise NotImplementedError


class PluginRegistry:
    """Registry for plugins."""

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: Dict[PluginType, Dict[str, Type[Plugin]]] = {}
        for plugin_type in PluginType:
            self._plugins[plugin_type] = {}
        self._lock = threading.RLock()
        logger.debug("PluginRegistry initialized")

    def register_plugin(
        self, plugin_class: Type[Plugin], mode: RegistrationMode = RegistrationMode.WARN
    ) -> None:
        """
        Register a plugin class with configurable behavior for duplicates.

        Args:
        ----
            plugin_class: The plugin class to register
            mode: How to handle duplicate registrations

        """
        # Create a temporary instance to access property values
        try:
            temp_instance = plugin_class()
            plugin_type = temp_instance.plugin_type
            plugin_name = temp_instance.name

            with self._lock:
                # Check if plugin already exists
                if plugin_name in self._plugins.get(plugin_type, {}):
                    if mode == RegistrationMode.STRICT:
                        msg = f"Plugin {plugin_name} of type {plugin_type} already exists"
                        raise ValueError(msg)
                    elif mode == RegistrationMode.WARN:
                        logger.warning(
                            f"Plugin {plugin_name} of type {plugin_type} already exists, overwriting"
                        )
                    elif mode == RegistrationMode.SKIP:
                        logger.debug(
                            f"Plugin {plugin_name} of type {plugin_type} already exists, skipping"
                        )
                        return

                # Register the plugin
                self._plugins[plugin_type][plugin_name] = plugin_class
                logger.debug(f"Registered plugin {plugin_name} of type {plugin_type}")
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
            if mode == RegistrationMode.STRICT:
                raise

    def get_plugin(self, plugin_type: PluginType, plugin_name: str) -> Optional[Type[Plugin]]:
        """
        Get a plugin by type and name.

        Args:
        ----
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin

        Returns:
        -------
            Optional[Type[Plugin]]: The plugin class, or None if not found

        """
        with self._lock:
            return self._plugins.get(plugin_type, {}).get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, Type[Plugin]]:
        """
        Get all plugins of a specific type.

        Args:
        ----
            plugin_type: Type of the plugins to retrieve

        Returns:
        -------
            Dict[str, Type[Plugin]]: Dictionary of plugin name to plugin class

        """
        with self._lock:
            return self._plugins.get(plugin_type, {}).copy()
