from __future__ import annotations

"""
Registry API module for Saplings.

This module provides the public API for registry and service locator components.
"""

from typing import Any, List, Optional

from saplings.api.stability import stable
from saplings.registry._internal import (
    IndexerRegistry as _IndexerRegistry,
)
from saplings.registry._internal import (
    PluginRegistry as _PluginRegistry,
)
from saplings.registry._internal import (
    PluginType as _PluginType,
)
from saplings.registry._internal import (
    RegistryContext as _RegistryContext,
)
from saplings.registry._internal import (
    ServiceLocator as _ServiceLocator,
)
from saplings.registry._internal import (
    discover_plugins as _discover_plugins,
)
from saplings.registry._internal import (
    get_plugin_registry as _get_plugin_registry,
)
from saplings.registry._internal import (
    register_plugin as _register_plugin,
)


@stable
class PluginRegistry(_PluginRegistry):
    """
    Registry for plugins in the Saplings framework.

    This class provides a registry for plugins, allowing them to be registered
    and retrieved by type and name.
    """


# Re-export the PluginType enum directly
PluginType = stable(_PluginType)


@stable
class RegistryContext(_RegistryContext):
    """
    Context for registry operations in the Saplings framework.

    This class provides a context for registry operations, allowing them to be
    scoped to a specific context.
    """


@stable
class ServiceLocator(_ServiceLocator):
    """
    Service locator for the Saplings framework.

    This class provides a service locator for resolving services and dependencies
    in the Saplings framework.
    """


@stable
class IndexerRegistry(_IndexerRegistry):
    """
    Registry for indexers in the Saplings framework.

    This class provides a registry for indexers, allowing them to be registered
    and retrieved by name.
    """


# Re-export the Plugin class
from saplings.registry._internal import Plugin as _Plugin

Plugin = stable(_Plugin)


# Re-export the RegistrationMode enum
from saplings.registry._internal import RegistrationMode as _RegistrationMode

RegistrationMode = stable(_RegistrationMode)


@stable
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
    result = _get_plugin_registry(registry)
    if isinstance(result, PluginRegistry):
        return result
    return PluginRegistry()


@stable
def register_plugin(
    plugin_class: Any, registry: Optional[PluginRegistry] = None, mode: Any = None
) -> None:
    """
    Register a plugin.

    Args:
    ----
        plugin_class: The plugin class to register
        registry: Optional registry to use (uses global registry if None)
        mode: How to handle duplicate registrations

    """
    # Handle mode parameter
    if mode is None:
        mode = _RegistrationMode.WARN

    return _register_plugin(plugin_class, registry, mode)


@stable
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
    result = _discover_plugins(registry, discover_entry_points, discover_directories)
    if isinstance(result, PluginRegistry):
        return result
    return PluginRegistry()


__all__ = [
    # Registry classes
    "PluginRegistry",
    "RegistryContext",
    "ServiceLocator",
    "IndexerRegistry",
    # Plugin types and modes
    "PluginType",
    "RegistrationMode",
    "Plugin",
    # Registry functions
    "get_plugin_registry",
    "register_plugin",
    "discover_plugins",
]
