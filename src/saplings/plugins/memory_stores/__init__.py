from __future__ import annotations

"""Memory store plugins for Saplings."""


from saplings.api.registry import RegistrationMode, register_plugin
from saplings.plugins.memory_stores.secure_memory_store import SecureMemoryStore

# Register the plugins with SKIP mode to avoid warnings
register_plugin(SecureMemoryStore, mode=RegistrationMode.SKIP)

__all__ = ["SecureMemoryStore"]
