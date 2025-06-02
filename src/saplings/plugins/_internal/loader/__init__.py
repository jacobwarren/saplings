from __future__ import annotations

"""
Loader module for plugin components.

This module provides plugin loading functionality for the Saplings framework.
"""

from saplings.plugins._internal.loader.plugin_loader import (
    PluginLoader,
    PluginLoadError,
    register_all_plugins,
)

__all__ = [
    "PluginLoader",
    "PluginLoadError",
    "register_all_plugins",
]
