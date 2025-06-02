from __future__ import annotations

"""
Cache API module for Saplings.

This module provides the main API for the unified caching system.
"""

from saplings.core._internal.caching.api import (
    cached,
    cached_property,
    clear_all_caches,
    clear_cache,
    get_cache,
    get_cache_stats,
    get_provider,
    register_provider,
)

__all__ = [
    "cached",
    "cached_property",
    "clear_all_caches",
    "clear_cache",
    "get_cache",
    "get_cache_stats",
    "get_provider",
    "register_provider",
]
