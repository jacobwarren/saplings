from __future__ import annotations

"""
Registry functions for Saplings.

This module provides registry functions for Saplings.
"""

import importlib
import logging
import pkgutil
import sys
import threading
from typing import List, Optional, Type

try:
    from importlib_metadata import entry_points

    ENTRY_POINTS_AVAILABLE = True
except ImportError:
    ENTRY_POINTS_AVAILABLE = False

from saplings.registry._internal.core.plugin import (
    Plugin,
    PluginRegistry,
    PluginType,
    RegistrationMode,
)

logger = logging.getLogger(__name__)

# Global registry instance for early initialization
_global_registry: Optional[PluginRegistry] = None
_registry_lock = threading.RLock()


def get_plugin_registry(registry: Optional[PluginRegistry] = None) -> PluginRegistry:
    """
    Get the plugin registry.

    This function supports both service locator and direct injection approaches.
    It ensures that a registry is always available, even during early initialization.

    Args:
    ----
        registry: Optional explicit registry to use

    Returns:
    -------
        PluginRegistry: The plugin registry

    """
    global _global_registry

    # If an explicit registry is provided, use it
    if registry is not None:
        return registry

    # Use thread-safe singleton pattern for global registry
    with _registry_lock:
        # First, try to get from global registry (fastest path)
        if _global_registry is not None:
            return _global_registry

        # Next, try to get from service locator
        # Use lazy import to avoid circular dependencies
        service_locator = None
        try:
            # Import the ServiceLocator class lazily
            from saplings.registry._internal.common.locator import get_service_locator

            service_locator = get_service_locator()
            registry = service_locator.get_service("plugin_registry")
            if registry is not None:
                _global_registry = registry
                return registry
        except Exception:
            # If service locator fails, we'll create a new registry below
            logger.debug("Failed to get registry from service locator, creating new registry")

        # Create a new registry and register it
        registry = PluginRegistry()
        _global_registry = registry

        # Try to register with service locator
        if service_locator is not None:
            try:
                service_locator.register_service("plugin_registry", registry)
                logger.debug("Registered plugin registry with service locator")
            except Exception:
                logger.debug("Failed to register plugin registry with service locator")

        return registry


def register_plugin(
    plugin_class: Type[Plugin],
    registry: Optional[PluginRegistry] = None,
    mode: RegistrationMode = RegistrationMode.WARN,
) -> None:
    """
    Register a plugin.

    Args:
    ----
        plugin_class: The plugin class to register
        registry: Optional registry to use (uses global registry if None)
        mode: How to handle duplicate registrations

    """
    plugin_registry = get_plugin_registry(registry)
    plugin_registry.register_plugin(plugin_class, mode)


def discover_plugins(
    registry: Optional[PluginRegistry] = None,
    discover_entry_points: bool = True,
    discover_directories: Optional[List[str]] = None,
) -> PluginRegistry:
    """
    Discover and register plugins from entry points and directories.

    This function looks for plugins in:
    1. Entry points (if discover_entry_points is True)
    2. Specified directories (if provided)

    Entry point groups:
    - saplings.model_adapters
    - saplings.memory_stores
    - saplings.validators
    - saplings.indexers
    - saplings.tools

    Args:
    ----
        registry: Optional registry to use (uses global registry if None)
        discover_entry_points: Whether to discover plugins from entry points
        discover_directories: Optional list of directories to search for plugins

    Returns:
    -------
        PluginRegistry: The registry with discovered plugins

    """
    # Get or create the registry
    plugin_registry = get_plugin_registry(registry)

    # Discover plugins from entry points
    if discover_entry_points and ENTRY_POINTS_AVAILABLE:
        try:
            # Map entry point groups to plugin types
            entry_point_groups = {
                "saplings.model_adapters": PluginType.MODEL_ADAPTER,
                "saplings.memory_stores": PluginType.MEMORY_STORE,
                "saplings.validators": PluginType.VALIDATOR,
                "saplings.indexers": PluginType.INDEXER,
                "saplings.tools": PluginType.TOOL,
            }

            # Discover plugins from each entry point group
            for group_name, plugin_type in entry_point_groups.items():
                try:
                    # Get entry points for this group
                    eps = entry_points(group=group_name)

                    # Load each entry point
                    for entry_point in eps:
                        try:
                            # Load the plugin class
                            plugin_class = entry_point.load()

                            # Register the plugin
                            plugin_registry.register_plugin(
                                plugin_class, mode=RegistrationMode.SILENT
                            )
                            logger.debug(f"Discovered plugin {entry_point.name} from {group_name}")
                        except Exception as e:
                            logger.warning(
                                f"Error loading plugin {entry_point.name} from {group_name}: {e}"
                            )
                except Exception as e:
                    logger.warning(f"Error discovering plugins from {group_name}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering plugins from entry points: {e}")

    # Discover plugins from directories
    if discover_directories:
        for directory in discover_directories:
            try:
                # Add the directory to the Python path
                original_path = list(sys.path)
                sys.path.insert(0, directory)

                try:
                    # Find all Python modules in the directory
                    for _, name, is_pkg in pkgutil.iter_modules([directory]):
                        try:
                            # Import the module
                            module = importlib.import_module(name)

                            # Find all classes in the module that are Plugin subclasses
                            for attr_name in dir(module):
                                try:
                                    attr = getattr(module, attr_name)

                                    # Check if it's a Plugin subclass
                                    if (
                                        isinstance(attr, type)
                                        and issubclass(attr, Plugin)
                                        and attr is not Plugin
                                    ):
                                        # Register the plugin
                                        plugin_registry.register_plugin(
                                            attr, mode=RegistrationMode.SILENT
                                        )
                                        logger.debug(
                                            f"Discovered plugin {attr_name} from {directory}"
                                        )
                                except Exception:
                                    # Skip attributes that can't be inspected
                                    pass
                        except Exception as e:
                            logger.warning(f"Error loading module {name} from {directory}: {e}")
                finally:
                    # Restore the original Python path
                    sys.path = original_path
            except Exception as e:
                logger.warning(f"Error discovering plugins from directory {directory}: {e}")

    logger.info("Plugin discovery completed")
    return plugin_registry
