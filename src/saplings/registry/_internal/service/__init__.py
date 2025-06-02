from __future__ import annotations

"""
Service module for registry components.

This module provides service locator functionality for the Saplings framework.
"""

from saplings.registry._internal.service.service_locator import (
    IndexerRegistry,
    ServiceLocator,
    get_plugin_registry,
)

__all__ = [
    "ServiceLocator",
    "IndexerRegistry",
    "get_plugin_registry",
]
