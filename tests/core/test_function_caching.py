"""
Tests for function caching.

This module provides tests for the function caching utilities in Saplings.
"""

import asyncio
import time
from typing import Dict, List, Optional

import pytest

from saplings.core.function_caching import (
    FunctionCache,
    FunctionCacheManager,
    cached,
    clear_all_caches,
    clear_cache,
)


class TestFunctionCache:
    """Test class for the function cache."""

    def test_get_set(self):
        """Test getting and setting values in the cache."""
        cache = FunctionCache(max_size=10, ttl=60)

        # Set a value
        cache.set("key1", "value1")

        # Get the value
        assert cache.get("key1") == "value1"

        # Get a nonexistent value
        assert cache.get("nonexistent") is None

    def test_expiration(self):
        """Test that values expire after TTL."""
        cache = FunctionCache(max_size=10, ttl=0.1)  # 100ms TTL

        # Set a value
        cache.set("key1", "value1")

        # Get the value immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.2)

        # Value should be expired
        assert cache.get("key1") is None

    def test_max_size(self):
        """Test that the cache respects max_size."""
        cache = FunctionCache(max_size=2, ttl=None)

        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 to make it most recently used
        assert cache.get("key1") == "value1"

        # Add a third value, which should evict key2
        cache.set("key3", "value3")

        # Check that key1 and key3 are in the cache, but key2 is not
        assert cache.get("key1") == "value1"
        assert cache.get("key3") == "value3"
        assert cache.get("key2") is None

    def test_delete(self):
        """Test deleting values from the cache."""
        cache = FunctionCache(max_size=10, ttl=60)

        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Delete a value
        assert cache.delete("key1") is True

        # Check that it's gone
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        # Delete a nonexistent value
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        """Test clearing the cache."""
        cache = FunctionCache(max_size=10, ttl=60)

        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Clear the cache
        cache.clear()

        # Check that all values are gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestFunctionCacheManager:
    """Test class for the function cache manager."""

    def test_get_cache(self):
        """Test getting a cache by namespace."""
        manager = FunctionCacheManager()

        # Get a cache
        cache1 = manager.get_cache("namespace1")

        # Get the same cache again
        cache2 = manager.get_cache("namespace1")

        # Get a different cache
        cache3 = manager.get_cache("namespace2")

        # Check that cache1 and cache2 are the same instance
        assert cache1 is cache2

        # Check that cache1 and cache3 are different instances
        assert cache1 is not cache3

    def test_clear_cache(self):
        """Test clearing a cache by namespace."""
        manager = FunctionCacheManager()

        # Get caches
        cache1 = manager.get_cache("namespace1")
        cache2 = manager.get_cache("namespace2")

        # Set values
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")

        # Clear one cache
        manager.clear_cache("namespace1")

        # Check that cache1 is cleared but cache2 is not
        assert cache1.get("key1") is None
        assert cache2.get("key2") == "value2"

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        manager = FunctionCacheManager()

        # Get caches
        cache1 = manager.get_cache("namespace1")
        cache2 = manager.get_cache("namespace2")

        # Set values
        cache1.set("key1", "value1")
        cache2.set("key2", "value2")

        # Clear all caches
        manager.clear_all_caches()

        # Check that both caches are cleared
        assert cache1.get("key1") is None
        assert cache2.get("key2") is None


class TestCachedDecorator:
    """Test class for the cached decorator."""

    def test_cached_function(self):
        """Test caching a synchronous function."""
        call_count = 0

        @cached(ttl=60)
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        # First call (cache miss)
        result1 = add(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same arguments (cache hit)
        result2 = add(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again

        # Call with different arguments (cache miss)
        result3 = add(3, 4)
        assert result3 == 7
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_async_function(self):
        """Test caching an asynchronous function."""
        call_count = 0

        @cached(ttl=60)
        async def add_async(a, b):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return a + b

        # First call (cache miss)
        result1 = await add_async(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same arguments (cache hit)
        result2 = await add_async(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again

        # Call with different arguments (cache miss)
        result3 = await add_async(3, 4)
        assert result3 == 7
        assert call_count == 2

    def test_cached_with_custom_namespace(self):
        """Test caching with a custom namespace."""
        call_count = 0

        @cached(namespace="custom")
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        # First call (cache miss)
        result1 = add(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Clear the cache
        clear_cache("custom")

        # Second call (cache miss due to clearing)
        result2 = add(1, 2)
        assert result2 == 3
        assert call_count == 2

    def test_cached_with_custom_key_generator(self):
        """Test caching with a custom key generator."""
        call_count = 0

        def key_generator(*args, **kwargs):
            return f"custom_key_{args[0]}_{kwargs.get('b', 0)}"

        @cached(key_generator=key_generator)
        def add(a, b=0):
            nonlocal call_count
            call_count += 1
            return a + b

        # First call (cache miss)
        result1 = add(1, b=2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same key (cache hit)
        result2 = add(1, b=2)
        assert result2 == 3
        assert call_count == 1

        # Call with different key (cache miss)
        result3 = add(3, b=4)
        assert result3 == 7
        assert call_count == 2

    def test_clear_cache_functions(self):
        """Test the clear_cache and clear_all_caches functions."""
        call_count1 = 0
        call_count2 = 0

        @cached(namespace="ns1")
        def func1():
            nonlocal call_count1
            call_count1 += 1
            return "result1"

        @cached(namespace="ns2")
        def func2():
            nonlocal call_count2
            call_count2 += 1
            return "result2"

        # First calls (cache misses)
        func1()
        func2()
        assert call_count1 == 1
        assert call_count2 == 1

        # Clear one cache
        clear_cache("ns1")

        # Second calls
        func1()  # Cache miss due to clearing
        func2()  # Cache hit
        assert call_count1 == 2
        assert call_count2 == 1

        # Clear all caches
        clear_all_caches()

        # Third calls (cache misses due to clearing)
        func1()
        func2()
        assert call_count1 == 3
        assert call_count2 == 2
