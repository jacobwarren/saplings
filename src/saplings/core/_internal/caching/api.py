from __future__ import annotations

"""
Cache API module for Saplings.

This module provides the main API for the unified caching system.
"""


import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar

from saplings.core._internal.caching.interface import (
    Cache,
    CacheProvider,
    CacheStats,
)
from saplings.core._internal.caching.keys import KeyBuilder
from saplings.core._internal.caching.memory import memory_cache_provider
from saplings.core.caching.interface import (
    CacheConfig,
    CacheStrategy,
)

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")

# Registry of cache providers
_providers: dict[str, CacheProvider] = {
    "memory": memory_cache_provider,
}


def register_provider(name: str, provider: CacheProvider) -> None:
    """
    Register a cache provider.

    Args:
    ----
        name: Name of the provider
        provider: Cache provider instance

    """
    _providers[name] = provider
    logger.info(f"Registered cache provider: {name}")


def get_provider(name: str) -> CacheProvider:
    """
    Get a cache provider by name.

    Args:
    ----
        name: Name of the provider

    Returns:
    -------
        CacheProvider: The provider

    Raises:
    ------
        ValueError: If the provider is not found

    """
    if name not in _providers:
        msg = f"Cache provider not found: {name}"
        raise ValueError(msg)
    return _providers[name]


def get_cache(
    namespace: str = "default",
    provider: str = "memory",
    max_size: int = 1000,
    ttl: int | None = 3600,
    strategy: CacheStrategy = CacheStrategy.LRU,
    **kwargs,
) -> Cache:
    """
    Get a cache by namespace and provider.

    Args:
    ----
        namespace: Namespace for the cache
        provider: Name of the provider
        max_size: Maximum number of items in the cache
        ttl: Time to live in seconds (None for no expiration)
        strategy: Cache eviction strategy
        **kwargs: Additional provider-specific options

    Returns:
    -------
        Cache: The cache

    """
    return get_provider(provider).get_cache(
        namespace=namespace, max_size=max_size, ttl=ttl, strategy=strategy, **kwargs
    )


def clear_cache(namespace: str = "default", provider: str = "memory") -> None:
    """
    Clear a cache by namespace and provider.

    Args:
    ----
        namespace: Namespace for the cache
        provider: Name of the provider

    """
    get_provider(provider).clear_cache(namespace)
    logger.info(f"Cleared cache: {namespace} (provider: {provider})")


def clear_all_caches(provider: str | None = None) -> None:
    """
    Clear all caches for a provider or all providers.

    Args:
    ----
        provider: Name of the provider (None for all providers)

    """
    if provider is None:
        for p in _providers.values():
            p.clear_all_caches()
        logger.info("Cleared all caches for all providers")
    else:
        get_provider(provider).clear_all_caches()
        logger.info(f"Cleared all caches for provider: {provider}")


def get_cache_stats(
    namespace: str | None = None, provider: str = "memory"
) -> dict[str, CacheStats]:
    """
    Get cache statistics.

    Args:
    ----
        namespace: Namespace for the cache (None for all caches)
        provider: Name of the provider

    Returns:
    -------
        Dict[str, CacheStats]: Cache statistics

    """
    p = get_provider(provider)

    if namespace:
        cache = p.get_cache(namespace=namespace)
        cache_stats = cache.get_stats()
        return {namespace: cache_stats} if cache_stats is not None else {}
    all_stats = p.get_all_stats()
    return all_stats if all_stats is not None else {}


def cached(
    namespace: str = "default",
    provider: str = "memory",
    ttl: int | None = 3600,
    strategy: CacheStrategy = CacheStrategy.LRU,
    key_builder: Callable | None = None,
    key_builder_kwargs: dict[str, Any] | None = None,
    **provider_kwargs,
) -> Callable[[Callable], Callable]:
    """
    Decorator for caching function results.

    This decorator can be applied to any function to cache its results.

    Args:
    ----
        namespace: Namespace for the cache
        provider: Name of the provider
        ttl: Time to live in seconds (None for default TTL)
        strategy: Cache eviction strategy
        key_builder: Function to build the cache key (None for default)
        key_builder_kwargs: Additional kwargs for the key builder
        **provider_kwargs: Additional provider-specific options

    Returns:
    -------
        Callable: Decorated function

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Get the cache
            cache = get_cache(
                namespace=namespace,
                provider=provider,
                ttl=ttl,
                strategy=strategy,
                **provider_kwargs,
            )

            # Build a cache key
            if key_builder:
                # Use custom key builder if provided
                key = key_builder(*args, **kwargs, **(key_builder_kwargs or {}))
            else:
                # Use default key builder
                key = KeyBuilder.build_function_key(
                    func_name=func.__qualname__, args=args, kwargs=kwargs
                )

            # Check if the result is in the cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__qualname__}")
                return cached_result

            # Call the function
            result = func(*args, **kwargs)

            # Cache the result
            cache.set(key, result, ttl=ttl)
            logger.debug(f"Cache miss for {func.__qualname__}, cached result")

            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get the cache
            cache = get_cache(
                namespace=namespace,
                provider=provider,
                ttl=ttl,
                strategy=strategy,
                **provider_kwargs,
            )

            # Build a cache key
            if key_builder:
                # Use custom key builder if provided
                key = key_builder(*args, **kwargs, **(key_builder_kwargs or {}))
            else:
                # Use default key builder
                key = KeyBuilder.build_function_key(
                    func_name=func.__qualname__, args=args, kwargs=kwargs
                )

            # Check if the result is in the cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__qualname__}")
                return cached_result

            # Call the function
            result = await func(*args, **kwargs)

            # Cache the result
            cache.set(key, result, ttl=ttl)
            logger.debug(f"Cache miss for {func.__qualname__}, cached result")

            return result

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def cached_property(
    namespace: str = "default",
    provider: str = "memory",
    ttl: int | None = 3600,
    strategy: CacheStrategy = CacheStrategy.LRU,
    **provider_kwargs,
) -> Callable[[Callable[[Any], T]], property]:
    """
    Decorator for caching properties.

    This decorator can be applied to a property to cache its value.
    It creates a property that caches the result of the getter function.

    Args:
    ----
        namespace: Namespace for the cache
        provider: Name of the provider
        ttl: Time to live in seconds (None for default TTL)
        strategy: Cache eviction strategy
        **provider_kwargs: Additional provider-specific options

    Returns:
    -------
        Callable: Decorated property

    """

    def decorator(func: Callable[[Any], T]) -> property:
        # Build a unique namespace for the property
        prop_namespace = f"{namespace}.{func.__qualname__}"

        @functools.wraps(func)
        def cached_getter(self: Any) -> T:
            # Get the cache
            cache = get_cache(
                namespace=prop_namespace,
                provider=provider,
                ttl=ttl,
                strategy=strategy,
                **provider_kwargs,
            )

            # Use the instance's ID as the cache key
            key = str(id(self))

            # Check if the value is in the cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Call the getter
            value = func(self)

            # Cache the value
            cache.set(key, value, ttl=ttl)

            return value

        return property(cached_getter)

    return decorator


# Export key classes and enums
__all__ = [
    "Cache",
    "CacheConfig",
    "CacheProvider",
    "CacheStats",
    "CacheStrategy",
    "KeyBuilder",
    "cached",
    "cached_property",
    "clear_all_caches",
    "clear_cache",
    "get_cache",
    "get_cache_stats",
    "get_provider",
    "register_provider",
]
