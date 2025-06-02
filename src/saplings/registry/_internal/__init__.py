from __future__ import annotations

"""
Internal implementation of the Registry module.

This module contains internal implementation details that are not part of the public API.
These components should not be used directly by application code.
"""

# Import from subdirectories
from saplings.registry._internal.context import (
    RegistryContext,
)
from saplings.registry._internal.core import (
    Plugin,
    PluginRegistry,
    PluginType,
    RegistrationMode,
    discover_plugins,
    get_plugin_registry,
    register_plugin,
)
from saplings.registry._internal.service import (
    IndexerRegistry,
    ServiceLocator,
)

__all__ = [
    # Core registry
    "PluginType",
    "RegistrationMode",
    "Plugin",
    "PluginRegistry",
    "register_plugin",
    "discover_plugins",
    "get_plugin_registry",
    # Registry context
    "RegistryContext",
    # Service locator
    "ServiceLocator",
    "IndexerRegistry",
]
