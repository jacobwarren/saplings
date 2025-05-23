from __future__ import annotations

"""Plugin modules for the Saplings framework."""

# Import the register_all_plugins function

from saplings.plugins.register import register_all_plugins

# Import plugins to make them available
try:
    # Import the plugin classes
    from saplings.plugins.indexers.code_indexer import CodeIndexer
    from saplings.plugins.memory_stores.secure_memory_store import SecureMemoryStore
    from saplings.plugins.validators.code_validator import CodeValidator
    from saplings.plugins.validators.factual_validator import FactualValidator

    # Register the plugins automatically
    register_all_plugins()
except ImportError:
    pass

__all__ = [
    "CodeIndexer",
    "CodeValidator",
    "FactualValidator",
    "SecureMemoryStore",
    "register_all_plugins",
]
