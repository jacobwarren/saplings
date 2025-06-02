from __future__ import annotations

"""
Internal module for plugin components.

This module provides the implementation of plugin components for the Saplings framework.
"""

# Import from subdirectories
from saplings.plugins._internal.config import (
    RegistrationMode,
)
from saplings.plugins._internal.core import (
    Plugin,
)
from saplings.plugins._internal.loader import (
    PluginLoader,
    PluginLoadError,
    register_all_plugins,
)
from saplings.plugins._internal.manager import (
    PluginRegistry,
)

# Import plugin implementations for convenience
from saplings.plugins.indexers.code_indexer import CodeIndexer
from saplings.plugins.memory_stores.secure_memory_store import SecureMemoryStore
from saplings.plugins.validators.code_validator import CodeValidator
from saplings.plugins.validators.factual_validator import FactualValidator

__all__ = [
    # Core
    "Plugin",
    # Manager
    "PluginRegistry",
    # Configuration
    "RegistrationMode",
    # Loader
    "PluginLoader",
    "PluginLoadError",
    "register_all_plugins",
    # Plugin implementations
    "CodeIndexer",
    "CodeValidator",
    "FactualValidator",
    "SecureMemoryStore",
]
