"""Memory store plugins for Saplings."""

from saplings.core.plugin import register_plugin
from saplings.plugins.memory_stores.secure_memory_store import SecureMemoryStore

# Register the plugins
register_plugin(SecureMemoryStore)

__all__ = ["SecureMemoryStore"]
