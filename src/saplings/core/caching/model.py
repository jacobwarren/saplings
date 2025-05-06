from __future__ import annotations

"""
Model caching adapter for Saplings.

This module provides adapters for caching model responses.
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.core.caching.api import cached, get_cache
from saplings.core.caching.interface import CacheStrategy
from saplings.core.caching.keys import KeyBuilder

if TYPE_CHECKING:
    from saplings.core.model_adapter import LLMResponse

logger = logging.getLogger(__name__)


def generate_with_cache(
    generate_func,
    cache_key: str,
    prompt: str | list[dict[str, Any]],
    namespace: str = "model",
    ttl: int | None = 3600,
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **kwargs,
) -> LLMResponse:
    """
    Generate a response with caching.

    Args:
    ----
        generate_func: Original generate function
        cache_key: Key identifying the model (provider:model_name)
        prompt: Prompt for the model
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **kwargs: Additional parameters for the generate function

    Returns:
    -------
        LLMResponse: The generated response

    """
    # Get the cache
    cache = get_cache(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
    )

    # Build a cache key
    key = KeyBuilder.build_model_key(cache_key=cache_key, prompt=prompt, **kwargs)

    # Check if the response is in the cache
    cached_response = cache.get(key)
    if cached_response is not None:
        logger.debug(f"Cache hit for model: {cache_key}")
        return cached_response

    # Generate the response
    response = generate_func(prompt, **kwargs)

    # Cache the response
    cache.set(key, response, ttl=ttl)
    logger.debug(f"Cache miss for model: {cache_key}, cached response")

    return response


async def generate_with_cache_async(
    generate_func,
    cache_key: str,
    prompt: str | list[dict[str, Any]],
    namespace: str = "model",
    ttl: int | None = 3600,
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **kwargs,
) -> LLMResponse:
    """
    Generate a response with caching asynchronously.

    Args:
    ----
        generate_func: Original generate function (async)
        cache_key: Key identifying the model (provider:model_name)
        prompt: Prompt for the model
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **kwargs: Additional parameters for the generate function

    Returns:
    -------
        LLMResponse: The generated response

    """
    # Get the cache
    cache = get_cache(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
    )

    # Build a cache key
    key = KeyBuilder.build_model_key(cache_key=cache_key, prompt=prompt, **kwargs)

    # Check if the response is in the cache
    cached_response = cache.get(key)
    if cached_response is not None:
        logger.debug(f"Cache hit for model: {cache_key}")
        return cached_response

    # Generate the response
    response = await generate_func(prompt, **kwargs)

    # Cache the response
    cache.set(key, response, ttl=ttl)
    logger.debug(f"Cache miss for model: {cache_key}, cached response")

    return response


def cached_model_response(
    namespace: str = "model",
    ttl: int | None = 3600,
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **provider_kwargs,
):
    """
    Decorator for caching model responses.

    This decorator can be applied to methods that generate model responses,
    such as the `generate` method of model adapters.

    Args:
    ----
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **provider_kwargs: Additional provider-specific options

    Returns:
    -------
        Callable: Decorated function

    """

    def key_builder(self, prompt: str, **kwargs):
        """Custom key builder for model responses."""
        # Extract provider and model_name from the instance
        provider = getattr(self, "provider", "unknown")
        model_name = getattr(self, "model_name", "unknown")
        cache_key = f"{provider}:{model_name}"

        return KeyBuilder.build_model_key(cache_key=cache_key, prompt=prompt, **kwargs)

    # Use the generic cached decorator with our custom key builder
    return cached(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
        key_builder=key_builder,
        **provider_kwargs,
    )
