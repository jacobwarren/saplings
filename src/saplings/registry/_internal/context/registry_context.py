from __future__ import annotations

"""
Registry context for Saplings.

This module provides the registry context for Saplings.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saplings.registry._internal.core.plugin import PluginRegistry


class RegistryContext:
    """Context for registry operations."""

    def __init__(self, registry: "PluginRegistry") -> None:
        """Initialize the registry context."""
        self.registry = registry
