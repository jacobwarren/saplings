"""
Model caching module for Saplings.

This module provides utilities for caching model responses to improve performance,
reduce costs, and enhance reliability of the agent framework.
"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from saplings.core.model_adapter import LLMResponse

logger = logging.getLogger(__name__)


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
    created_at: datetime = Field(default_factory=datetime.now, description="When the cache was created")
    last_hit_at: Optional[datetime] = Field(None, description="When the cache was last hit")
    last_miss_at: Optional[datetime] = Field(None, description="When the cache was last missed")

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        result = self.model_dump()
        result["hit_rate"] = self.hit_rate
        return result


class ModelCache:
    """
    Cache for model responses with multiple eviction strategies and persistence.

    This cache implements several eviction strategies and can optionally persist
    to disk to survive process restarts.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[int] = 3600,
        namespace: str = "default",
        strategy: CacheStrategy = CacheStrategy.LRU,
        persist: bool = False,
        persist_path: Optional[str] = None
    ):
        """
        Initialize the model cache.

        Args:
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds (None for no expiration)
            namespace: Namespace for the cache
            strategy: Cache eviction strategy
            persist: Whether to persist the cache to disk
            persist_path: Path to persist the cache (default: ~/.saplings/cache)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.namespace = namespace
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
        self.stats = CacheStats(max_size=max_size)

    def _init_cache(self) -> None:
        """Initialize the cache data structures."""
        self._cache: Dict[str, Tuple[LLMResponse, float]] = {}
        self._access_times: Dict[str, float] = {}  # For LRU
        self._access_counts: Dict[str, int] = {}  # For LFU
        self._insertion_order: List[str] = []  # For FIFO

    def _load_from_disk(self) -> None:
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

                logger.info(f"Loaded {len(self._cache)} items from cache: {self.namespace}")
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            self._init_cache()

    def _save_to_disk(self) -> None:
        """Save the cache to disk."""
        if not self.persist:
            return

        try:
            data = {
                "cache": self._cache,
                "access_times": self._access_times,
                "access_counts": self._access_counts,
                "insertion_order": self._insertion_order
            }

            with open(self.cache_file, "wb") as f:
                pickle.dump(data, f)

            logger.debug(f"Saved {len(self._cache)} items to cache: {self.namespace}")
        except (IOError, pickle.PickleError) as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def _clean_expired(self) -> None:
        """Clean up expired items."""
        if self.ttl is None:
            return

        now = time.time()
        expired_keys = [
            key for key, (_, expiration) in self._cache.items()
            if now > expiration
        ]

        for key in expired_keys:
            self.delete(key)

    def get(self, key: str) -> Optional[LLMResponse]:
        """
        Get a response from the cache.

        Args:
            key: Cache key

        Returns:
            Optional[LLMResponse]: Cached response or None if not found or expired
        """
        # Clean expired items periodically
        if self.ttl is not None and time.time() % 10 < 0.1:  # ~10% chance to clean
            self._clean_expired()

        if key not in self._cache:
            self.stats.misses += 1
            self.stats.last_miss_at = datetime.now()
            return None

        response, expiration = self._cache[key]

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

        logger.debug(f"Cache hit for key: {key}")
        return response

    def _update_access_metadata(self, key: str) -> None:
        """Update access metadata for a key based on the strategy."""
        now = time.time()

        # Update LRU data
        self._access_times[key] = now

        # Update LFU data
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

    def set(self, key: str, response: LLMResponse) -> None:
        """
        Set a response in the cache.

        Args:
            key: Cache key
            response: Response to cache
        """
        # Check if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Evict based on strategy
            self._evict_item()

        # Calculate expiration time
        expiration = time.time() + self.ttl if self.ttl is not None else float("inf")

        # Store response
        self._cache[key] = (response, expiration)

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

        logger.debug(f"Cache set for key: {key}")

    def _evict_item(self) -> None:
        """Evict an item based on the cache strategy."""
        if not self._cache:
            return

        key_to_evict = None

        if self.strategy == CacheStrategy.LRU:
            # Find the least recently used key
            if self._access_times:
                key_to_evict = min(self._access_times, key=self._access_times.get)

        elif self.strategy == CacheStrategy.LFU:
            # Find the least frequently used key
            if self._access_counts:
                key_to_evict = min(self._access_counts, key=self._access_counts.get)

        elif self.strategy == CacheStrategy.FIFO:
            # Get the oldest key
            if self._insertion_order:
                key_to_evict = self._insertion_order[0]

        if key_to_evict:
            self.delete(key_to_evict)
            self.stats.evictions += 1
            logger.debug(f"Cache evicted item with key: {key_to_evict} using {self.strategy} strategy")

    def delete(self, key: str) -> bool:
        """
        Delete a response from the cache.

        Args:
            key: Cache key

        Returns:
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

            logger.debug(f"Cache delete for key: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_times.clear()
        self._access_counts.clear()
        self._insertion_order.clear()

        # Reset stats
        self.stats = CacheStats(max_size=self.max_size)

        # Save to disk if persistence is enabled
        if self.persist:
            self._save_to_disk()

        logger.debug(f"Cache cleared for namespace: {self.namespace}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        self.stats.size = len(self._cache)
        return self.stats.to_dict()


class ModelCacheManager:
    """
    Manager for model caches.

    This class manages multiple cache instances with different namespaces
    and provides global operations like clearing all caches.
    """

    def __init__(self):
        """Initialize the model cache manager."""
        self._caches: Dict[str, ModelCache] = {}

    def get_cache(
        self,
        namespace: str = "default",
        max_size: int = 1000,
        ttl: Optional[int] = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        persist: bool = False,
        persist_path: Optional[str] = None
    ) -> ModelCache:
        """
        Get a cache by namespace.

        Args:
            namespace: Namespace for the cache
            max_size: Maximum number of items in the cache
            ttl: Time to live in seconds (None for no expiration)
            strategy: Cache eviction strategy
            persist: Whether to persist the cache to disk
            persist_path: Path to persist the cache

        Returns:
            ModelCache: The cache
        """
        if namespace not in self._caches:
            self._caches[namespace] = ModelCache(
                max_size=max_size,
                ttl=ttl,
                namespace=namespace,
                strategy=strategy,
                persist=persist,
                persist_path=persist_path
            )
        return self._caches[namespace]

    def clear_cache(self, namespace: str = "default") -> None:
        """
        Clear a cache by namespace.

        Args:
            namespace: Namespace for the cache
        """
        if namespace in self._caches:
            self._caches[namespace].clear()

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dict[str, Dict[str, Any]]: Statistics for all caches
        """
        return {
            namespace: cache.get_stats()
            for namespace, cache in self._caches.items()
        }


# Create a singleton instance
cache_manager = ModelCacheManager()


def generate_cache_key(
    model_uri: str,
    prompt: Union[str, List[Dict[str, Any]]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> str:
    """
    Generate a cache key for a model request.

    Args:
        model_uri: URI of the model
        prompt: Prompt for the model
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        **kwargs: Additional arguments for generation

    Returns:
        str: Cache key
    """
    # Convert prompt to a string if it's a list of messages
    if isinstance(prompt, list):
        try:
            prompt_str = json.dumps(prompt, sort_keys=True)
        except (TypeError, ValueError):
            # If the prompt can't be serialized, use its string representation
            prompt_str = str(prompt)
    else:
        prompt_str = prompt

    # Create a dictionary of all parameters
    params = {
        "model_uri": model_uri,
        "prompt": prompt_str,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Add additional kwargs
    for key, value in kwargs.items():
        # Skip functions and function_call as they might not be serializable
        if key in ["functions", "function_call"]:
            if key == "functions" and isinstance(value, list):
                # For functions, just use the function names
                try:
                    params[key] = [f.get("name", str(i)) for i, f in enumerate(value)]
                except (AttributeError, TypeError):
                    params[key] = str(value)
            elif key == "function_call" and isinstance(value, dict):
                # For function_call, just use the name
                params[key] = value.get("name", str(value))
            else:
                params[key] = str(value)
        else:
            # For other parameters, use the value directly if it's a simple type
            if isinstance(value, (str, int, float, bool, type(None))):
                params[key] = value
            else:
                # For complex types, use their string representation
                params[key] = str(value)

    # Serialize the parameters to a string
    params_str = json.dumps(params, sort_keys=True)

    # Hash the string to create a cache key
    return hashlib.md5(params_str.encode()).hexdigest()


def get_model_cache(
    namespace: str = "default",
    max_size: int = 1000,
    ttl: Optional[int] = 3600,
    strategy: CacheStrategy = CacheStrategy.LRU,
    persist: bool = False,
    persist_path: Optional[str] = None
) -> ModelCache:
    """
    Get a model cache by namespace.

    Args:
        namespace: Namespace for the cache
        max_size: Maximum number of items in the cache
        ttl: Time to live in seconds (None for no expiration)
        strategy: Cache eviction strategy
        persist: Whether to persist the cache to disk
        persist_path: Path to persist the cache

    Returns:
        ModelCache: The cache
    """
    return cache_manager.get_cache(
        namespace=namespace,
        max_size=max_size,
        ttl=ttl,
        strategy=strategy,
        persist=persist,
        persist_path=persist_path
    )


def clear_model_cache(namespace: str = "default") -> None:
    """
    Clear a model cache by namespace.

    Args:
        namespace: Namespace for the cache
    """
    cache_manager.clear_cache(namespace)


def clear_all_model_caches() -> None:
    """Clear all model caches."""
    cache_manager.clear_all_caches()


def get_cache_stats(namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Get cache statistics.

    Args:
        namespace: Namespace for the cache (None for all caches)

    Returns:
        Dict[str, Any]: Cache statistics
    """
    if namespace:
        cache = get_model_cache(namespace)
        return cache.get_stats()
    else:
        return cache_manager.get_all_stats()


def cached_model_response(
    namespace: str = "default",
    ttl: Optional[int] = 3600,
    strategy: CacheStrategy = CacheStrategy.LRU,
    persist: bool = False,
    persist_path: Optional[str] = None
):
    """
    Decorator for caching model responses.

    This decorator can be applied to methods that generate model responses,
    such as the `generate` method of model adapters.

    Args:
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        strategy: Cache eviction strategy
        persist: Whether to persist the cache to disk
        persist_path: Path to persist the cache

    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Get the cache
            cache = get_model_cache(
                namespace=namespace,
                ttl=ttl,
                strategy=strategy,
                persist=persist,
                persist_path=persist_path
            )

            # Generate a cache key
            model_uri = getattr(self, "model_uri", "unknown")
            prompt = args[0] if args else kwargs.get("prompt", "")

            # Extract other parameters
            params = {}
            if "max_tokens" in kwargs:
                params["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]

            # Add remaining kwargs
            for k, v in kwargs.items():
                if k not in ["prompt", "max_tokens", "temperature"]:
                    params[k] = v

            cache_key = generate_cache_key(
                model_uri=model_uri,
                prompt=prompt,
                **params
            )

            # Check if the response is in the cache
            cached_response = cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Cache hit for {model_uri}")
                return cached_response

            # Generate the response
            response = await func(self, *args, **kwargs)

            # Cache the response
            cache.set(cache_key, response)

            return response

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Get the cache
            cache = get_model_cache(
                namespace=namespace,
                ttl=ttl,
                strategy=strategy,
                persist=persist,
                persist_path=persist_path
            )

            # Generate a cache key
            model_uri = getattr(self, "model_uri", "unknown")
            prompt = args[0] if args else kwargs.get("prompt", "")

            # Extract other parameters
            params = {}
            if "max_tokens" in kwargs:
                params["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]

            # Add remaining kwargs
            for k, v in kwargs.items():
                if k not in ["prompt", "max_tokens", "temperature"]:
                    params[k] = v

            cache_key = generate_cache_key(
                model_uri=model_uri,
                prompt=prompt,
                **params
            )

            # Check if the response is in the cache
            cached_response = cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Cache hit for {model_uri}")
                return cached_response

            # Generate the response
            response = func(self, *args, **kwargs)

            # Cache the response
            cache.set(cache_key, response)

            return response

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
