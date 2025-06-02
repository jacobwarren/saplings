from __future__ import annotations

"""
Core registry module for registry components.

This module provides core registry functionality for the Saplings framework.
"""

from saplings.registry._internal.core.plugin import (
    Plugin,
    PluginRegistry,
    PluginType,
    RegistrationMode,
)
from saplings.registry._internal.core.registry import (
    discover_plugins,
    get_plugin_registry,
    register_plugin,
)

__all__ = [
    "PluginType",
    "RegistrationMode",
    "Plugin",
    "PluginRegistry",
    "register_plugin",
    "discover_plugins",
    "get_plugin_registry",
]
