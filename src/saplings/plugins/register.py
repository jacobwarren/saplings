"""
Utility module for manually registering Saplings plugins.

This module provides a function to register all built-in plugins
and can also be run as a script to register plugins manually.

Example:
    # Register all plugins
    from saplings.plugins.register import register_all_plugins
    register_all_plugins()

    # Or run as a script
    python -m saplings.plugins.register
"""

from saplings.core.plugin import register_plugin
from saplings.plugins.indexers.code_indexer import CodeIndexer
from saplings.plugins.memory_stores.secure_memory_store import SecureMemoryStore
from saplings.plugins.validators.code_validator import CodeValidator
from saplings.plugins.validators.factual_validator import FactualValidator


def register_all_plugins():
    """Register all plugins."""
    # Register memory store plugins
    register_plugin(SecureMemoryStore)

    # Register validator plugins
    register_plugin(CodeValidator)
    register_plugin(FactualValidator)

    # Register indexer plugins
    register_plugin(CodeIndexer)

    print("All plugins registered successfully!")


if __name__ == "__main__":
    register_all_plugins()
