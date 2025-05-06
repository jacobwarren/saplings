from __future__ import annotations

"""
saplings.core.caching.
=====================

This module provides a unified caching system for Saplings, with pluggable backends,
consistent key generation, TTL, and eviction policies.
"""

# Import main API components

from saplings.core.caching.api import (
    Cache,
    CacheConfig,
    CacheProvider,
    CacheStats,
    CacheStrategy,
    KeyBuilder,
    cached,
    cached_property,
    clear_all_caches,
    clear_cache,
    get_cache,
    get_cache_stats,
    get_provider,
    register_provider,
)

# Import model caching components
from saplings.core.caching.model import (
    cached_model_response,
    generate_with_cache,
    generate_with_cache_async,
)

# Import vector caching components
from saplings.core.caching.vector import (
    cached_embedding,
    cached_retrieval,
    embed_with_cache,
    embed_with_cache_async,
    retrieve_with_cache,
    retrieve_with_cache_async,
)

# Define what's exposed via import *
__all__ = [
    # Core interfaces
    "Cache",
    "CacheConfig",
    "CacheProvider",
    "CacheStats",
    "CacheStrategy",
    "KeyBuilder",
    # Core API
    "cached",
    # Vector caching
    "cached_embedding",
    # Model caching
    "cached_model_response",
    "cached_property",
    "cached_retrieval",
    "clear_all_caches",
    "clear_cache",
    "embed_with_cache",
    "embed_with_cache_async",
    "generate_with_cache",
    "generate_with_cache_async",
    "get_cache",
    "get_cache_stats",
    "get_provider",
    "register_provider",
    "retrieve_with_cache",
    "retrieve_with_cache_async",
]
