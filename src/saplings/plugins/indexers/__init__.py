"""Indexer plugins for Saplings."""

from saplings.core.plugin import register_plugin
from saplings.plugins.indexers.code_indexer import CodeIndexer

# Register the plugin
register_plugin(CodeIndexer)

__all__ = ["CodeIndexer"]
