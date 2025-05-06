from __future__ import annotations

"""
Memory-based cache implementation for Saplings.

This module provides an in-memory cache implementation.
"""


import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from saplings.core.caching.interface import Cache, CacheProvider, CacheStats, CacheStrategy

K = TypeVar("K")
T = TypeVar("T")


class MemoryCache(Cache[K, T]):
    """
    In-memory cache implementation with multiple eviction strategies and persistence.

    This cache implements several eviction strategies and can optionally persist
    to disk to survive process restarts.
    """

    def __init__(
        self,
        namespace: str = "default",
        max_size: int = 1000,
        ttl: int | None = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        persist: bool = False,
        persist_path: str | None = None,
    ) -> None:
        """
        Initialize the memory cache.

        Args:
        ----
            namespace: Namespace for the cache
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds (None for no expiration)
            strategy: Cache eviction strategy
            persist: Whether to persist the cache to disk
            persist_path: Path to persist the cache (default: ~/.saplings/cache)

        """
        self.namespace = namespace
        self.max_size = max_size
        self.ttl = ttl
        self.strategy = strategy
        self.persist = persist

        # Set up persistence path
        if persist:
            if persist_path:
                self.persist_path = Path(persist_path)
            else:
                self.persist_path = Path.home() / ".saplings" / "cache"

            # Create directory if it doesn't exist
            os.makedirs(self.persist_path, exist_ok=True)

            # Full path to the cache file
            self.cache_file = self.persist_path / f"{namespace}.cache"

            # Load from disk if it exists
            if self.cache_file.exists():
                self._load_from_disk()
            else:
                self._init_cache()
        else:
            self._init_cache()

        # Initialize stats
        self.stats = CacheStats(
            max_size=max_size,
            hits=0,
            misses=0,
            evictions=0,
            size=0,
            last_hit_at=None,
            last_miss_at=None,
        )

    def _init_cache(self):
        """Initialize the cache data structures."""
        self._cache: dict[K, tuple[T, float]] = {}
        self._access_times: dict[K, float] = {}  # For LRU
        self._access_counts: dict[K, int] = {}  # For LFU
        self._insertion_order: list[K] = []  # For FIFO

    def _load_from_disk(self):
        """Load the cache from disk."""
        try:
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)

                self._cache = data.get("cache", {})
                self._access_times = data.get("access_times", {})
                self._access_counts = data.get("access_counts", {})
                self._insertion_order = data.get("insertion_order", [])

                # Clean up expired items
                self._clean_expired()
        except (FileNotFoundError, pickle.PickleError, EOFError):
            # If loading fails, initialize an empty cache
            self._init_cache()

    def _save_to_disk(self):
        """Save the cache to disk."""
        if not self.persist:
            return

        try:
            data = {
                "cache": self._cache,
                "access_times": self._access_times,
                "access_counts": self._access_counts,
                "insertion_order": self._insertion_order,
            }

            with open(self.cache_file, "wb") as f:
                pickle.dump(data, f)
        except (OSError, pickle.PickleError):
            # If saving fails, just continue (the cache is still in memory)
            pass

    def _clean_expired(self):
        """Clean up expired items."""
        if self.ttl is None:
            return

        now = time.time()
        expired_keys = [key for key, (_, expiration) in self._cache.items() if now > expiration]

        for key in expired_keys:
            self.delete(key)

    def get(self, key: K) -> T | None:
        """
        Get a value from the cache.

        Args:
        ----
            key: Cache key

        Returns:
        -------
            Optional[T]: Cached value or None if not found or expired

        """
        # Clean expired items periodically
        if self.ttl is not None and time.time() % 10 < 0.1:  # ~10% chance to clean
            self._clean_expired()

        if key not in self._cache:
            self.stats.misses += 1
            self.stats.last_miss_at = datetime.now()
            return None

        value, expiration = self._cache[key]

        # Check if expired
        if self.ttl is not None and time.time() > expiration:
            # Remove expired item
            self.delete(key)
            self.stats.misses += 1
            self.stats.last_miss_at = datetime.now()
            return None

        # Update access metadata based on strategy
        self._update_access_metadata(key)

        # Update stats
        self.stats.hits += 1
        self.stats.last_hit_at = datetime.now()

        return value

    def _update_access_metadata(self, key: K) -> None:
        """Update access metadata for a key based on the strategy."""
        now = time.time()

        # Update LRU data
        self._access_times[key] = now

        # Update LFU data
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

    def set(self, key: K, value: T, ttl: int | None = None) -> None:
        """
        Set a value in the cache.

        Args:
        ----
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default TTL)

        """
        # Check if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Evict based on strategy
            self._evict_item()

        # Calculate expiration time
        expiration = time.time() + (
            ttl if ttl is not None else self.ttl if self.ttl is not None else float("inf")
        )

        # Store value
        self._cache[key] = (value, expiration)

        # Update metadata based on strategy
        self._update_access_metadata(key)

        # For FIFO, add to insertion order if not already present
        if key not in self._insertion_order:
            self._insertion_order.append(key)

        # Update stats
        self.stats.size = len(self._cache)

        # Save to disk if persistence is enabled
        if self.persist:
            self._save_to_disk()

    def _evict_item(self):
        """Evict an item based on the cache strategy."""
        if not self._cache:
            return

        key_to_evict = None

        if self.strategy == CacheStrategy.LRU:
            # Find the least recently used key
            if self._access_times:
                key_to_evict = min(
                    self._access_times, key=lambda k: self._access_times.get(k, float("inf"))
                )

        elif self.strategy == CacheStrategy.LFU:
            # Find the least frequently used key
            if self._access_counts:
                key_to_evict = min(
                    self._access_counts, key=lambda k: self._access_counts.get(k, float("inf"))
                )

        elif self.strategy == CacheStrategy.FIFO:
            # Get the oldest key
            if self._insertion_order:
                key_to_evict = self._insertion_order[0]

        if key_to_evict:
            self.delete(key_to_evict)
            self.stats.evictions += 1

    def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.

        Args:
        ----
            key: Cache key

        Returns:
        -------
            bool: True if the key was deleted, False otherwise

        """
        if key in self._cache:
            del self._cache[key]

            # Clean up metadata
            if key in self._access_times:
                del self._access_times[key]

            if key in self._access_counts:
                del self._access_counts[key]

            if key in self._insertion_order:
                self._insertion_order.remove(key)

            # Update stats
            self.stats.size = len(self._cache)

            # Save to disk if persistence is enabled
            if self.persist:
                self._save_to_disk()

            return True
        return False

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_times.clear()
        self._access_counts.clear()
        self._insertion_order.clear()

        # Reset stats
        self.stats = CacheStats(
            max_size=self.max_size,
            hits=0,
            misses=0,
            evictions=0,
            size=0,
            last_hit_at=None,
            last_miss_at=None,
        )

        # Save to disk if persistence is enabled
        if self.persist:
            self._save_to_disk()

    def get_stats(self):
        """
        Get cache statistics.

        Returns
        -------
            CacheStats: Cache statistics

        """
        self.stats.size = len(self._cache)
        return self.stats


class MemoryCacheProvider(CacheProvider):
    """
    Memory-based cache provider.

    This provider manages multiple in-memory cache instances with different namespaces.
    """

    def __init__(self) -> None:
        """Initialize the memory cache provider."""
        self._caches: dict[str, MemoryCache] = {}

    def get_cache(
        self,
        namespace: str = "default",
        max_size: int = 1000,
        ttl: int | None = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        persist: bool = False,
        persist_path: str | None = None,
        **kwargs,
    ) -> Cache:
        """
        Get a cache by namespace.

        Args:
        ----
            namespace: Namespace for the cache
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds (None for no expiration)
            strategy: Cache eviction strategy
            persist: Whether to persist the cache to disk
            persist_path: Path to persist the cache
            **kwargs: Additional options (ignored for memory cache)

        Returns:
        -------
            Cache: The cache

        """
        if namespace not in self._caches:
            self._caches[namespace] = MemoryCache(
                namespace=namespace,
                max_size=max_size,
                ttl=ttl,
                strategy=strategy,
                persist=persist,
                persist_path=persist_path,
            )
        return self._caches[namespace]

    def clear_cache(self, namespace: str = "default") -> None:
        """
        Clear a cache by namespace.

        Args:
        ----
            namespace: Namespace for the cache

        """
        if namespace in self._caches:
            self._caches[namespace].clear()

    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()

    def get_all_stats(self):
        """
        Get statistics for all caches.

        Returns
        -------
            Dict[str, CacheStats]: Statistics for all caches

        """
        return {namespace: cache.get_stats() for namespace, cache in self._caches.items()}


# Create a singleton instance
memory_cache_provider = MemoryCacheProvider()
