from __future__ import annotations

"""
Cache interface module for Saplings.

This module defines the interfaces for the unified caching system.
"""


import abc
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")
K = TypeVar("K")


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


class CacheStats(BaseModel):
    """Statistics for a cache."""

    hits: int = Field(0, description="Number of cache hits")
    misses: int = Field(0, description="Number of cache misses")
    evictions: int = Field(0, description="Number of cache evictions")
    size: int = Field(0, description="Current number of items in the cache")
    max_size: int = Field(0, description="Maximum number of items in the cache")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the cache was created"
    )
    last_hit_at: datetime | None = Field(None, description="When the cache was last hit")
    last_miss_at: datetime | None = Field(None, description="When the cache was last missed")

    @property
    def hit_rate(self):
        """Calculate the hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary."""
        result = self.model_dump()
        result["hit_rate"] = self.hit_rate
        return result


class Cache(Generic[K, T], abc.ABC):
    """
    Abstract base class for cache implementations.

    This interface defines the methods that all cache implementations must provide.
    Caches are generic over key type K and value type T.
    """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def set(self, key: K, value: T, ttl: int | None = None) -> None:
        """
        Set a value in the cache.

        Args:
        ----
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default TTL)

        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def clear(self):
        """Clear the cache."""

    @abc.abstractmethod
    def get_stats(self):
        """
        Get cache statistics.

        Returns
        -------
            CacheStats: Cache statistics

        """


class CacheProvider(abc.ABC):
    """
    Abstract base class for cache providers.

    Cache providers are responsible for creating and managing cache instances.
    """

    @abc.abstractmethod
    def get_cache(
        self,
        namespace: str = "default",
        max_size: int = 1000,
        ttl: int | None = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
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
            **kwargs: Additional provider-specific options

        Returns:
        -------
            Cache: The cache

        """

    @abc.abstractmethod
    def clear_cache(self, namespace: str = "default") -> None:
        """
        Clear a cache by namespace.

        Args:
        ----
            namespace: Namespace for the cache

        """

    @abc.abstractmethod
    def clear_all_caches(self):
        """Clear all caches."""

    @abc.abstractmethod
    def get_all_stats(self):
        """
        Get statistics for all caches.

        Returns
        -------
            Dict[str, CacheStats]: Statistics for all caches

        """


class CacheConfig(BaseModel):
    """Base configuration for caches."""

    provider: str = Field("memory", description="Cache provider to use (memory, redis, etc.)")
    namespace: str = Field("default", description="Namespace for the cache")
    max_size: int = Field(1000, description="Maximum number of items in the cache")
    ttl: int | None = Field(3600, description="Time to live in seconds (None for no expiration)")
    strategy: CacheStrategy = Field(CacheStrategy.LRU, description="Cache eviction strategy")
    persist: bool = Field(False, description="Whether to persist the cache")
    persist_path: str | None = Field(None, description="Path to persist the cache")
