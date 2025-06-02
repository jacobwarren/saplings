from __future__ import annotations

"""
Caching interface for Saplings.

This module provides interfaces for caching operations.
"""

from saplings.core._internal.caching.interface import (
    CacheConfig,
    CacheStrategy,
)

# Forward declarations for circular imports
ICacheService = CacheConfig  # Using CacheConfig as a placeholder
MemoryCacheStrategy = CacheStrategy.LRU  # Updated to match internal implementation
NoCacheStrategy = CacheStrategy.FIFO  # Updated to match internal implementation
PersistentCacheStrategy = CacheStrategy.LFU  # Updated to match internal implementation

__all__ = [
    "CacheConfig",
    "CacheStrategy",
    "ICacheService",
    "MemoryCacheStrategy",
    "NoCacheStrategy",
    "PersistentCacheStrategy",
]
