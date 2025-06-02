from __future__ import annotations

"""
Plugin system for Saplings.

This module provides the plugin system for Saplings, allowing for
extensibility through plugins.
"""

from saplings.core._internal.plugin import (
    Plugin,
    PluginRegistry,
    PluginType,
    RegistrationMode,
    discover_plugins,
    get_plugin_registry,
    register_plugin,
)

__all__ = [
    "Plugin",
    "PluginRegistry",
    "PluginType",
    "RegistrationMode",
    "discover_plugins",
    "get_plugin_registry",
    "register_plugin",
]
