from __future__ import annotations

"""Indexer plugins for Saplings."""


from saplings.api.registry import RegistrationMode, register_plugin
from saplings.plugins.indexers.code_indexer import CodeIndexer

# Register the plugin with SKIP mode to avoid warnings
register_plugin(CodeIndexer, mode=RegistrationMode.SKIP)

__all__ = ["CodeIndexer"]
