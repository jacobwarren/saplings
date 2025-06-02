from __future__ import annotations

"""
Registry module for Saplings.

This module re-exports the public API from saplings.api.registry.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides registry and service locator functionality for Saplings.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.registry.

__all__ = [
    "IndexerRegistry",
    "Plugin",
    "PluginRegistry",
    "PluginType",
    "RegistrationMode",
    "RegistryContext",
    "ServiceLocator",
    "discover_plugins",
    "get_plugin_registry",
    "register_plugin",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.registry import (
            IndexerRegistry,
            Plugin,
            PluginRegistry,
            PluginType,
            RegistrationMode,
            RegistryContext,
            ServiceLocator,
            discover_plugins,
            get_plugin_registry,
            register_plugin,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "IndexerRegistry": IndexerRegistry,
            "Plugin": Plugin,
            "PluginRegistry": PluginRegistry,
            "PluginType": PluginType,
            "RegistrationMode": RegistrationMode,
            "RegistryContext": RegistryContext,
            "ServiceLocator": ServiceLocator,
            "discover_plugins": discover_plugins,
            "get_plugin_registry": get_plugin_registry,
            "register_plugin": register_plugin,
        }

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
