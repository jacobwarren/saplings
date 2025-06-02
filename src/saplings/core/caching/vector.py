from __future__ import annotations

"""
Vector caching module for Saplings.

This module provides caching utilities for vector operations.
"""

from saplings.core._internal.caching.vector import (
    cached_embedding,
    cached_retrieval,
    embed_with_cache,
    embed_with_cache_async,
    retrieve_with_cache,
    retrieve_with_cache_async,
)

__all__ = [
    "cached_embedding",
    "cached_retrieval",
    "embed_with_cache",
    "embed_with_cache_async",
    "retrieve_with_cache",
    "retrieve_with_cache_async",
]
