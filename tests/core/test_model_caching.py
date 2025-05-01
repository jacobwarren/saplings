"""
Tests for model caching.

This module provides tests for the model caching utilities in Saplings.
"""

import asyncio
import tempfile
import time
from typing import Dict

import pytest

from saplings.core.model_adapter import LLMResponse, ModelURI
from saplings.core.model_caching import (
    CacheStrategy,
    ModelCache,
    ModelCacheManager,
    cache_manager,
    cached_model_response,
    clear_all_model_caches,
    clear_model_cache,
    generate_cache_key,
    get_cache_stats,
    get_model_cache,
)


class TestModelCache:
    """Test class for the model cache."""

    def test_get_set(self):
        """Test getting and setting values in the cache."""
        cache = ModelCache(max_size=10, ttl=60)

        # Create a response
        response = LLMResponse(
            text="Test response",
            model_uri="openai://gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "openai"},
        )

        # Set a value
        cache.set("key1", response)

        # Get the value
        cached_response = cache.get("key1")
        assert cached_response is not None
        assert cached_response.text == "Test response"
        assert cached_response.model_uri == "openai://gpt-4"
        assert cached_response.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

        # Get a nonexistent value
        assert cache.get("nonexistent") is None

        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_expiration(self):
        """Test that values expire after TTL."""
        cache = ModelCache(max_size=10, ttl=0.1)  # 100ms TTL

        # Create a response
        response = LLMResponse(text="Test response", model_uri="openai://gpt-4")

        # Set a value
        cache.set("key1", response)

        # Get the value immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(0.2)

        # Value should be expired
        assert cache.get("key1") is None

    def test_max_size_lru(self):
        """Test that the cache respects max_size with LRU strategy."""
        cache = ModelCache(max_size=2, ttl=None, strategy=CacheStrategy.LRU)

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")
        response3 = LLMResponse(text="Response 3", model_uri="openai://gpt-4")

        # Set values
        cache.set("key1", response1)
        cache.set("key2", response2)

        # Access key1 to make it most recently used
        assert cache.get("key1") is not None

        # Add a third value, which should evict key2
        cache.set("key3", response3)

        # Check that key1 and key3 are in the cache, but key2 is not
        assert cache.get("key1") is not None
        assert cache.get("key3") is not None
        assert cache.get("key2") is None

        # Check stats
        stats = cache.get_stats()
        assert stats["evictions"] >= 1
        assert stats["size"] == 2

    def test_max_size_lfu(self):
        """Test that the cache respects max_size with LFU strategy."""
        cache = ModelCache(max_size=2, ttl=None, strategy=CacheStrategy.LFU)

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")
        response3 = LLMResponse(text="Response 3", model_uri="openai://gpt-4")

        # Set values
        cache.set("key1", response1)
        cache.set("key2", response2)

        # Access key1 multiple times to increase its frequency
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        # Add a third value, which should evict key2 (least frequently used)
        cache.set("key3", response3)

        # Check that key1 and key3 are in the cache, but key2 is not
        assert cache.get("key1") is not None
        assert cache.get("key3") is not None
        assert cache.get("key2") is None

    def test_max_size_fifo(self):
        """Test that the cache respects max_size with FIFO strategy."""
        cache = ModelCache(max_size=2, ttl=None, strategy=CacheStrategy.FIFO)

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")
        response3 = LLMResponse(text="Response 3", model_uri="openai://gpt-4")

        # Set values
        cache.set("key1", response1)
        cache.set("key2", response2)

        # Access key1 to make it most recently used (shouldn't matter for FIFO)
        cache.get("key1")
        cache.get("key1")

        # Add a third value, which should evict key1 (first in)
        cache.set("key3", response3)

        # Check that key2 and key3 are in the cache, but key1 is not
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_delete(self):
        """Test deleting values from the cache."""
        cache = ModelCache(max_size=10, ttl=60)

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")

        # Set values
        cache.set("key1", response1)
        cache.set("key2", response2)

        # Delete a value
        assert cache.delete("key1") is True

        # Check that it's gone
        assert cache.get("key1") is None
        assert cache.get("key2") is not None

        # Delete a nonexistent value
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        """Test clearing the cache."""
        cache = ModelCache(max_size=10, ttl=60)

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")

        # Set values
        cache.set("key1", response1)
        cache.set("key2", response2)

        # Clear the cache
        cache.clear()

        # Check that all values are gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestModelCacheManager:
    """Test class for the model cache manager."""

    def test_get_cache(self):
        """Test getting a cache by namespace."""
        manager = ModelCacheManager()

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
        manager = ModelCacheManager()

        # Get caches
        cache1 = manager.get_cache("namespace1")
        cache2 = manager.get_cache("namespace2")

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")

        # Set values
        cache1.set("key1", response1)
        cache2.set("key2", response2)

        # Clear one cache
        manager.clear_cache("namespace1")

        # Check that cache1 is cleared but cache2 is not
        assert cache1.get("key1") is None
        assert cache2.get("key2") is not None

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        manager = ModelCacheManager()

        # Get caches
        cache1 = manager.get_cache("namespace1")
        cache2 = manager.get_cache("namespace2")

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")

        # Set values
        cache1.set("key1", response1)
        cache2.set("key2", response2)

        # Clear all caches
        manager.clear_all_caches()

        # Check that both caches are cleared
        assert cache1.get("key1") is None
        assert cache2.get("key2") is None


class TestCacheKeyGeneration:
    """Test class for cache key generation."""

    def test_generate_cache_key_with_string_prompt(self):
        """Test generating a cache key with a string prompt."""
        # Generate a key
        key1 = generate_cache_key(
            model_uri="openai://gpt-4", prompt="Hello, world!", max_tokens=100, temperature=0.7
        )

        # Generate the same key again
        key2 = generate_cache_key(
            model_uri="openai://gpt-4", prompt="Hello, world!", max_tokens=100, temperature=0.7
        )

        # Generate a different key
        key3 = generate_cache_key(
            model_uri="openai://gpt-4", prompt="Different prompt", max_tokens=100, temperature=0.7
        )

        # Check that the keys are as expected
        assert key1 == key2
        assert key1 != key3

    def test_generate_cache_key_with_message_prompt(self):
        """Test generating a cache key with a message prompt."""
        # Create a message prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]

        # Generate a key
        key1 = generate_cache_key(
            model_uri="openai://gpt-4", prompt=messages, max_tokens=100, temperature=0.7
        )

        # Generate the same key again
        key2 = generate_cache_key(
            model_uri="openai://gpt-4", prompt=messages, max_tokens=100, temperature=0.7
        )

        # Generate a different key
        key3 = generate_cache_key(
            model_uri="openai://gpt-4",
            prompt=[{"role": "user", "content": "Different prompt"}],
            max_tokens=100,
            temperature=0.7,
        )

        # Check that the keys are as expected
        assert key1 == key2
        assert key1 != key3

    def test_generate_cache_key_with_functions(self):
        """Test generating a cache key with functions."""
        # Create a function
        function = {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
        }

        # Generate a key
        key1 = generate_cache_key(
            model_uri="openai://gpt-4",
            prompt="What's the weather like in San Francisco?",
            functions=[function],
            function_call="auto",
        )

        # Generate the same key again
        key2 = generate_cache_key(
            model_uri="openai://gpt-4",
            prompt="What's the weather like in San Francisco?",
            functions=[function],
            function_call="auto",
        )

        # Generate a different key
        key3 = generate_cache_key(
            model_uri="openai://gpt-4",
            prompt="What's the weather like in San Francisco?",
            functions=[function],
            function_call={"name": "get_weather"},
        )

        # Check that the keys are as expected
        assert key1 == key2
        assert key1 != key3

    def test_generate_cache_key_with_complex_kwargs(self):
        """Test generating a cache key with complex kwargs."""
        # Generate a key with complex kwargs
        key = generate_cache_key(
            model_uri="openai://gpt-4",
            prompt="Hello, world!",
            complex_object={"key": "value", "nested": {"key": "value"}},
            complex_list=[1, 2, 3, {"key": "value"}],
        )

        # Check that the key is a string
        assert isinstance(key, str)
        assert len(key) > 0


class TestPersistence:
    """Test class for cache persistence."""

    def test_persistence(self):
        """Test that the cache can be persisted to disk and loaded back."""
        # Create a temporary directory for the cache
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a cache with persistence
            cache1 = ModelCache(
                max_size=10, ttl=None, namespace="test_persist", persist=True, persist_path=temp_dir
            )

            # Create a response
            response = LLMResponse(
                text="Test response",
                model_uri="openai://gpt-4",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                metadata={"provider": "openai"},
            )

            # Set a value
            cache1.set("key1", response)

            # Create a new cache with the same namespace and path
            cache2 = ModelCache(
                max_size=10, ttl=None, namespace="test_persist", persist=True, persist_path=temp_dir
            )

            # Check that the value was loaded from disk
            cached_response = cache2.get("key1")
            assert cached_response is not None
            assert cached_response.text == "Test response"
            assert cached_response.model_uri == "openai://gpt-4"

            # Delete the value
            cache2.delete("key1")

            # Create a third cache to verify the deletion was persisted
            cache3 = ModelCache(
                max_size=10, ttl=None, namespace="test_persist", persist=True, persist_path=temp_dir
            )

            # Check that the value is gone
            assert cache3.get("key1") is None

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        # Create a cache with a specific namespace
        cache = ModelCache(max_size=10, ttl=None, namespace="stats_test")

        # Register the cache with the manager to ensure it's tracked
        cache_manager._caches["stats_test"] = cache

        # Create a response
        response = LLMResponse(text="Test response", model_uri="openai://gpt-4")

        # Set a value
        cache.set("key1", response)

        # Get the value (hit)
        cache.get("key1")

        # Try to get a nonexistent value (miss)
        cache.get("nonexistent")

        # Get stats
        stats = cache.get_stats()

        # Check stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["evictions"] == 0
        assert "hit_rate" in stats
        assert stats["hit_rate"] == 0.5  # 1 hit, 1 miss

        # Get stats for this specific namespace
        namespace_stats = get_cache_stats("stats_test")
        assert namespace_stats["hits"] == 1

        # Get global stats
        all_stats = get_cache_stats()
        assert isinstance(all_stats, dict)
        assert "stats_test" in all_stats


class TestCacheDecorator:
    """Test class for the cache decorator."""

    def test_sync_decorator(self):
        """Test the decorator with a synchronous function."""

        # Create a class with a method to decorate
        class TestModel:
            def __init__(self):
                self.model_uri = "test://model"
                self.call_count = 0

            @cached_model_response(namespace="test_decorator")
            def generate(self, prompt, **kwargs):
                self.call_count += 1
                return LLMResponse(text=f"Response to: {prompt}", model_uri=self.model_uri)

        # Create an instance
        model = TestModel()

        # Call the method
        response1 = model.generate("test prompt")
        assert response1.text == "Response to: test prompt"
        assert model.call_count == 1

        # Call again with the same prompt (should use cache)
        response2 = model.generate("test prompt")
        assert response2.text == "Response to: test prompt"
        assert model.call_count == 1  # Should not increment

        # Call with a different prompt
        response3 = model.generate("different prompt")
        assert response3.text == "Response to: different prompt"
        assert model.call_count == 2

        # Clear the cache
        clear_model_cache("test_decorator")

        # Call again with the first prompt (should miss cache)
        response4 = model.generate("test prompt")
        assert response4.text == "Response to: test prompt"
        assert model.call_count == 3

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test the decorator with an asynchronous function."""

        # Create a class with an async method to decorate
        class TestAsyncModel:
            def __init__(self):
                self.model_uri = "test://async_model"
                self.call_count = 0

            @cached_model_response(namespace="test_async_decorator")
            async def generate(self, prompt, **kwargs):
                self.call_count += 1
                await asyncio.sleep(0.01)  # Simulate async operation
                return LLMResponse(text=f"Async response to: {prompt}", model_uri=self.model_uri)

        # Create an instance
        model = TestAsyncModel()

        # Call the method
        response1 = await model.generate("test prompt")
        assert response1.text == "Async response to: test prompt"
        assert model.call_count == 1

        # Call again with the same prompt (should use cache)
        response2 = await model.generate("test prompt")
        assert response2.text == "Async response to: test prompt"
        assert model.call_count == 1  # Should not increment

        # Call with a different prompt
        response3 = await model.generate("different prompt")
        assert response3.text == "Async response to: different prompt"
        assert model.call_count == 2


class TestConvenienceFunctions:
    """Test class for the convenience functions."""

    def test_get_model_cache(self):
        """Test the get_model_cache function."""
        # Get a cache with default parameters
        cache1 = get_model_cache("test_namespace")

        # Get the same cache again
        cache2 = get_model_cache("test_namespace")

        # Check that they are the same instance
        assert cache1 is cache2

        # Get a cache with custom parameters
        with tempfile.TemporaryDirectory() as temp_dir:
            cache3 = get_model_cache(
                namespace="custom_namespace",
                max_size=100,
                ttl=3600,
                strategy=CacheStrategy.LFU,
                persist=True,
                persist_path=temp_dir,
            )

            # Verify the cache was created with the right parameters
            assert cache3.namespace == "custom_namespace"
            assert cache3.max_size == 100
            assert cache3.ttl == 3600
            assert cache3.strategy == CacheStrategy.LFU
            assert cache3.persist is True

    def test_clear_model_cache(self):
        """Test the clear_model_cache function."""
        # Get a cache
        cache = get_model_cache("test_namespace")

        # Create a response
        response = LLMResponse(text="Test response", model_uri="openai://gpt-4")

        # Set a value
        cache.set("key1", response)

        # Clear the cache
        clear_model_cache("test_namespace")

        # Check that the value is gone
        assert cache.get("key1") is None

    def test_clear_all_model_caches(self):
        """Test the clear_all_model_caches function."""
        # Get caches
        cache1 = get_model_cache("namespace1")
        cache2 = get_model_cache("namespace2")

        # Create responses
        response1 = LLMResponse(text="Response 1", model_uri="openai://gpt-4")
        response2 = LLMResponse(text="Response 2", model_uri="openai://gpt-4")

        # Set values
        cache1.set("key1", response1)
        cache2.set("key2", response2)

        # Clear all caches
        clear_all_model_caches()

        # Check that both caches are cleared
        assert cache1.get("key1") is None
        assert cache2.get("key2") is None
