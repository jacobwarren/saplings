from __future__ import annotations

"""
Vector caching adapter for Saplings.

This module provides adapters for caching vector store operations.
"""


import logging
from typing import TYPE_CHECKING

from saplings.core.caching.api import cached, get_cache
from saplings.core.caching.interface import CacheStrategy
from saplings.core.caching.keys import KeyBuilder

if TYPE_CHECKING:
    from saplings.memory.document import Document

logger = logging.getLogger(__name__)


def retrieve_with_cache(
    retrieve_func,
    collection: str,
    query: str | list[float],
    namespace: str = "vector",
    ttl: int | None = 3600,
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **kwargs,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents with caching.

    Args:
    ----
        retrieve_func: Original retrieve function
        collection: Name of the vector collection
        query: Query text or embedding
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **kwargs: Additional parameters for the retrieve function

    Returns:
    -------
        List[Tuple[Document, float]]: The retrieved documents with scores

    """
    # Get the cache
    cache = get_cache(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
    )

    # Build a cache key
    key = KeyBuilder.build_vector_key(collection=collection, query=query, **kwargs)

    # Check if the results are in the cache
    cached_results = cache.get(key)
    if cached_results is not None:
        logger.debug(f"Cache hit for vector query in collection: {collection}")
        return cached_results

    # Retrieve the documents
    results = retrieve_func(query, **kwargs)

    # Cache the results
    cache.set(key, results, ttl=ttl)
    logger.debug(f"Cache miss for vector query in collection: {collection}, cached results")

    return results


async def retrieve_with_cache_async(
    retrieve_func,
    collection: str,
    query: str | list[float],
    namespace: str = "vector",
    ttl: int | None = 3600,
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **kwargs,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents with caching asynchronously.

    Args:
    ----
        retrieve_func: Original retrieve function (async)
        collection: Name of the vector collection
        query: Query text or embedding
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **kwargs: Additional parameters for the retrieve function

    Returns:
    -------
        List[Tuple[Document, float]]: The retrieved documents with scores

    """
    # Get the cache
    cache = get_cache(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
    )

    # Build a cache key
    key = KeyBuilder.build_vector_key(collection=collection, query=query, **kwargs)

    # Check if the results are in the cache
    cached_results = cache.get(key)
    if cached_results is not None:
        logger.debug(f"Cache hit for vector query in collection: {collection}")
        return cached_results

    # Retrieve the documents
    results = await retrieve_func(query, **kwargs)

    # Cache the results
    cache.set(key, results, ttl=ttl)
    logger.debug(f"Cache miss for vector query in collection: {collection}, cached results")

    return results


def embed_with_cache(
    embed_func,
    collection: str,
    text: str | list[str],
    namespace: str = "vector",
    ttl: int | None = 86400,  # Longer TTL for embeddings (1 day)
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **kwargs,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings with caching.

    Args:
    ----
        embed_func: Original embed function
        collection: Name of the vector collection
        text: Text to embed
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **kwargs: Additional parameters for the embed function

    Returns:
    -------
        Union[List[float], List[List[float]]]: The generated embeddings

    """
    # Get the cache
    cache = get_cache(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
    )

    # Handle single text vs. list of texts
    if isinstance(text, str):
        # Build a cache key for single text
        key = KeyBuilder.build_vector_key(
            collection=collection, query=text, operation="embed", **kwargs
        )

        # Check if the embedding is in the cache
        cached_embedding = cache.get(key)
        if cached_embedding is not None:
            logger.debug(f"Cache hit for embedding in collection: {collection}")
            return cached_embedding

        # Generate the embedding
        embedding = embed_func(text, **kwargs)

        # Cache the embedding
        cache.set(key, embedding, ttl=ttl)
        logger.debug(f"Cache miss for embedding in collection: {collection}, cached embedding")

        return embedding
    # For lists, we need to check each item individually
    results = []
    cache_hits = 0

    for item in text:
        # Build a cache key for this item
        key = KeyBuilder.build_vector_key(
            collection=collection, query=item, operation="embed", **kwargs
        )

        # Check if the embedding is in the cache
        cached_embedding = cache.get(key)
        if cached_embedding is not None:
            results.append(cached_embedding)
            cache_hits += 1
        else:
            # We'll need to generate embeddings for missing items
            break

    # If we got all results from cache
    if cache_hits == len(text):
        logger.debug(f"Cache hit for all {len(text)} embeddings in collection: {collection}")
        return results

    # Otherwise, generate all embeddings (simpler than partial caching)
    embeddings = embed_func(text, **kwargs)

    # Cache each embedding
    for i, item in enumerate(text):
        key = KeyBuilder.build_vector_key(
            collection=collection, query=item, operation="embed", **kwargs
        )
        cache.set(key, embeddings[i], ttl=ttl)

    logger.debug(
        f"Cache miss for some embeddings in collection: {collection}, cached {len(text)} embeddings"
    )

    return embeddings


async def embed_with_cache_async(
    embed_func,
    collection: str,
    text: str | list[str],
    namespace: str = "vector",
    ttl: int | None = 86400,  # Longer TTL for embeddings (1 day)
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **kwargs,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings with caching asynchronously.

    Args:
    ----
        embed_func: Original embed function (async)
        collection: Name of the vector collection
        text: Text to embed
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **kwargs: Additional parameters for the embed function

    Returns:
    -------
        Union[List[float], List[List[float]]]: The generated embeddings

    """
    # Get the cache
    cache = get_cache(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
    )

    # Handle single text vs. list of texts
    if isinstance(text, str):
        # Build a cache key for single text
        key = KeyBuilder.build_vector_key(
            collection=collection, query=text, operation="embed", **kwargs
        )

        # Check if the embedding is in the cache
        cached_embedding = cache.get(key)
        if cached_embedding is not None:
            logger.debug(f"Cache hit for embedding in collection: {collection}")
            return cached_embedding

        # Generate the embedding
        embedding = await embed_func(text, **kwargs)

        # Cache the embedding
        cache.set(key, embedding, ttl=ttl)
        logger.debug(f"Cache miss for embedding in collection: {collection}, cached embedding")

        return embedding
    # For lists, we need to check each item individually
    results = []
    cache_hits = 0

    for item in text:
        # Build a cache key for this item
        key = KeyBuilder.build_vector_key(
            collection=collection, query=item, operation="embed", **kwargs
        )

        # Check if the embedding is in the cache
        cached_embedding = cache.get(key)
        if cached_embedding is not None:
            results.append(cached_embedding)
            cache_hits += 1
        else:
            # We'll need to generate embeddings for missing items
            break

    # If we got all results from cache
    if cache_hits == len(text):
        logger.debug(f"Cache hit for all {len(text)} embeddings in collection: {collection}")
        return results

    # Otherwise, generate all embeddings (simpler than partial caching)
    embeddings = await embed_func(text, **kwargs)

    # Cache each embedding
    for i, item in enumerate(text):
        key = KeyBuilder.build_vector_key(
            collection=collection, query=item, operation="embed", **kwargs
        )
        cache.set(key, embeddings[i], ttl=ttl)

    logger.debug(
        f"Cache miss for some embeddings in collection: {collection}, cached {len(text)} embeddings"
    )

    return embeddings


def cached_retrieval(
    collection_attr: str = "collection_name",
    namespace: str = "vector",
    ttl: int | None = 3600,
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **provider_kwargs,
):
    """
    Decorator for caching vector retrieval operations.

    This decorator can be applied to retrieval methods of vector stores.

    Args:
    ----
        collection_attr: Attribute name for the collection name
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **provider_kwargs: Additional provider-specific options

    Returns:
    -------
        Callable: Decorated function

    """

    def key_builder(self, query: str, **kwargs):
        """Custom key builder for retrieval operations."""
        # Extract collection name from the instance
        collection = getattr(self, collection_attr, "default")

        return KeyBuilder.build_vector_key(collection=collection, query=query, **kwargs)

    # Use the generic cached decorator with our custom key builder
    return cached(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
        key_builder=key_builder,
        **provider_kwargs,
    )


def cached_embedding(
    collection_attr: str = "collection_name",
    namespace: str = "vector",
    ttl: int | None = 86400,  # Longer TTL for embeddings (1 day)
    provider: str = "memory",
    strategy: CacheStrategy = CacheStrategy.LRU,
    **provider_kwargs,
):
    """
    Decorator for caching embedding operations.

    This decorator can be applied to embedding methods of vector stores.

    Args:
    ----
        collection_attr: Attribute name for the collection name
        namespace: Namespace for the cache
        ttl: Time to live in seconds (None for no expiration)
        provider: Cache provider to use
        strategy: Cache eviction strategy
        **provider_kwargs: Additional provider-specific options

    Returns:
    -------
        Callable: Decorated function

    """

    def key_builder(self, text: str, **kwargs):
        """Custom key builder for embedding operations."""
        # Extract collection name from the instance
        collection = getattr(self, collection_attr, "default")

        # Handle single text vs. list of texts
        if isinstance(text, str):
            return KeyBuilder.build_vector_key(
                collection=collection, query=text, operation="embed", **kwargs
            )
        # For lists, we'll just return a string indicating it's a batch
        # The actual caching is handled in the wrapper
        return f"batch_embed_{id(text)}"

    # Use the generic cached decorator with our custom key builder
    return cached(
        namespace=namespace,
        provider=provider,
        ttl=ttl,
        strategy=strategy,
        key_builder=key_builder,
        **provider_kwargs,
    )
