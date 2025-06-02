from __future__ import annotations

"""
Caching configuration module for Saplings retrieval.

This module defines the configuration classes for retrieval caching.
"""


from pydantic import BaseModel, Field

from saplings.core.caching.interface import CacheStrategy


class CacheConfig(BaseModel):
    """Configuration for retrieval caching."""

    enabled: bool = Field(True, description="Whether to enable caching")
    namespace: str = Field("vector", description="Namespace for the cache")
    ttl: int | None = Field(3600, description="Time to live in seconds (None for no expiration)")
    provider: str = Field("memory", description="Cache provider to use")
    strategy: CacheStrategy = Field(CacheStrategy.LRU, description="Cache eviction strategy")
    embedding_ttl: int | None = Field(
        86400, description="Time to live for embedding cache entries (1 day by default)"
    )
    retrieval_ttl: int | None = Field(
        3600, description="Time to live for retrieval cache entries (1 hour by default)"
    )
    max_size: int = Field(1000, description="Maximum number of items in the cache")
    persist: bool = Field(False, description="Whether to persist the cache")
    persist_path: str | None = Field(None, description="Path to persist the cache")

    @classmethod
    def default(cls):
        """
        Create a default configuration.

        Returns
        -------
            CacheConfig: Default configuration

        """
        return cls(
            enabled=True,
            namespace="vector",
            ttl=3600,
            provider="memory",
            strategy=CacheStrategy.LRU,
            embedding_ttl=86400,
            retrieval_ttl=3600,
            max_size=1000,
            persist=False,
            persist_path=None,
        )

    @classmethod
    def disabled(cls):
        """
        Create a configuration with caching disabled.

        Returns
        -------
            CacheConfig: Configuration with caching disabled

        """
        return cls(
            enabled=False,
            namespace="vector",
            ttl=3600,
            provider="memory",
            strategy=CacheStrategy.LRU,
            embedding_ttl=86400,
            retrieval_ttl=3600,
            max_size=1000,
            persist=False,
            persist_path=None,
        )

    @classmethod
    def high_performance(cls):
        """
        Create a configuration optimized for high performance.

        Returns
        -------
            CacheConfig: High performance configuration

        """
        return cls(
            enabled=True,
            namespace="vector",
            ttl=None,  # No expiration
            provider="memory",
            strategy=CacheStrategy.LFU,  # Least Frequently Used (good for skewed access patterns)
            embedding_ttl=None,  # No expiration for embeddings
            retrieval_ttl=7200,  # 2 hours for retrievals
            max_size=10000,  # Larger cache
            persist=True,  # Persist to disk
            persist_path="./cache/retrieval",  # Default path for persistence
        )

    @classmethod
    def memory_efficient(cls):
        """
        Create a configuration optimized for memory efficiency.

        Returns
        -------
            CacheConfig: Memory efficient configuration

        """
        return cls(
            enabled=True,
            namespace="vector",
            ttl=1800,  # 30 minutes
            provider="memory",
            strategy=CacheStrategy.LRU,  # Least Recently Used (good for general use)
            embedding_ttl=43200,  # 12 hours for embeddings
            retrieval_ttl=1800,  # 30 minutes for retrievals
            max_size=500,  # Smaller cache
            persist=False,  # Don't persist to disk
            persist_path=None,  # No persistence path needed
        )
