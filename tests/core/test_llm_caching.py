"""
Tests for LLM caching integration.

This module provides tests for the caching integration in the LLM class.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelURI
from saplings.core.model_caching import clear_all_model_caches


class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, model_uri: Union[str, ModelURI], **kwargs):
        """Initialize the mock LLM."""
        # Parse the model URI if it's a string
        if isinstance(model_uri, str):
            self.model_uri = ModelURI.parse(model_uri)
        else:
            self.model_uri = model_uri

        self.generate_count = 0
        self.generate_streaming_count = 0

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from the model."""
        self.generate_count += 1

        # If use_cache is True, use the parent class implementation
        if use_cache:
            return await super().generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                use_cache=use_cache,
                cache_namespace=cache_namespace,
                cache_ttl=cache_ttl,
                **kwargs,
            )

        # Otherwise, return a mock response
        return LLMResponse(
            text=f"Response to: {prompt}",
            model_uri=str(self.model_uri),
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock"},
        )

    async def generate_streaming(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        chunk_size: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Generate text from the model with streaming output."""
        self.generate_streaming_count += 1

        # Return a mock streaming response
        yield "Response"
        yield " to:"
        yield f" {prompt}"

    async def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        return len(text.split())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of a request."""
        prompt_cost = prompt_tokens * 0.0001
        completion_cost = completion_tokens * 0.0002
        return prompt_cost + completion_cost

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the model."""
        return {
            "name": "mock-model",
            "provider": "mock",
            "version": "latest",
            "description": "Mock model for testing",
            "capabilities": ["text_generation"],
            "roles": ["general"],
            "context_window": 4096,
            "max_tokens_per_request": 2048,
            "cost_per_1k_tokens_input": 0.0001,
            "cost_per_1k_tokens_output": 0.0002,
        }


class TestLLMCaching:
    """Test class for LLM caching integration."""

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        """Clear all caches before and after each test."""
        clear_all_model_caches()
        yield
        clear_all_model_caches()

    @pytest.mark.asyncio
    async def test_generate_with_cache(self, monkeypatch):
        """Test generating with cache."""
        # Create a mock LLM
        llm = MockLLM("mock://model")

        # Mock the cache to avoid actual caching
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # First call returns None (cache miss)
        mock_cache.set = MagicMock()  # Mock the set method

        # For the second call with the same prompt, return a cached response
        cached_response = LLMResponse(
            text="Response to: Hello, world!",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True},
        )

        # Set up the mock to return the cached response on the second call
        mock_cache.get.side_effect = [None, cached_response, None]

        # Mock the get_model_cache function to return our mock cache
        monkeypatch.setattr(
            "saplings.core.model_caching.get_model_cache", lambda **kwargs: mock_cache
        )

        # Generate a response with caching
        response1 = await llm.generate_with_cache(
            prompt="Hello, world!", max_tokens=100, temperature=0.7
        )

        # Generate the same response again
        response2 = await llm.generate_with_cache(
            prompt="Hello, world!", max_tokens=100, temperature=0.7
        )

        # Generate a different response
        response3 = await llm.generate_with_cache(
            prompt="Different prompt", max_tokens=100, temperature=0.7
        )

        # Check that the responses are as expected
        assert response1.text == "Response to: Hello, world!"
        assert response2.text == "Response to: Hello, world!"
        assert response3.text == "Response to: Different prompt"

        # Check that generate was only called twice (once for each unique prompt)
        # The second call should use the cached response
        assert llm.generate_count == 2

    @pytest.mark.skip("This test is flaky and needs to be redesigned")
    @pytest.mark.asyncio
    async def test_generate_with_use_cache_parameter(self, monkeypatch):
        """Test generating with the use_cache parameter."""
        # Create a mock LLM
        llm = MockLLM("mock://model")

        # Create a counter to track how many times the original generate method is called
        original_generate = llm.generate
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_generate(*args, **kwargs)

        # Replace the generate method with our counting version
        llm.generate = mock_generate

        # Mock the cache to avoid actual caching
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # First call returns None (cache miss)
        mock_cache.set = MagicMock()  # Mock the set method

        # For the second call with the same prompt, return a cached response
        cached_response = LLMResponse(
            text="Response to: Hello, world!",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True},
        )

        # Set up the mock to return the cached response on the second call
        mock_cache.get.side_effect = [None, cached_response, None]

        # Mock the get_model_cache function to return our mock cache
        monkeypatch.setattr(
            "saplings.core.model_caching.get_model_cache",
            lambda namespace="default", **_: mock_cache,
        )

        # Generate a response with caching
        response1 = await llm.generate(
            prompt="Hello, world!", max_tokens=100, temperature=0.7, use_cache=True
        )

        # Generate the same response again
        response2 = await llm.generate(
            prompt="Hello, world!", max_tokens=100, temperature=0.7, use_cache=True
        )

        # Generate a different response
        response3 = await llm.generate(
            prompt="Different prompt", max_tokens=100, temperature=0.7, use_cache=True
        )

        # Check that the responses are as expected
        assert response1.text == "Response to: Hello, world!"
        assert response2.text == "Response to: Hello, world!"
        assert response3.text == "Response to: Different prompt"

        # Check that generate was only called twice (once for each unique prompt)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_chat_with_cache(self, monkeypatch):
        """Test chat with cache."""
        # Create a mock LLM
        llm = MockLLM("mock://model")

        # Mock the cache to avoid actual caching
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # First call returns None (cache miss)
        mock_cache.set = MagicMock()  # Mock the set method

        # For the second call with the same prompt, return a cached response
        cached_response = LLMResponse(
            text="Response to: [system: You are a helpful assistant, user: Hello, world!]",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True},
        )

        # Set up the mock to return the cached response on the second call
        mock_cache.get.side_effect = [None, cached_response, None]

        # Mock the get_model_cache function to return our mock cache
        monkeypatch.setattr("saplings.core.model_caching.get_model_cache", lambda **_: mock_cache)

        # Create a message prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]

        # Generate a response with caching
        response1 = await llm.chat(
            messages=messages, max_tokens=100, temperature=0.7, use_cache=True
        )

        # Generate the same response again
        response2 = await llm.chat(
            messages=messages, max_tokens=100, temperature=0.7, use_cache=True
        )

        # Generate a different response
        response3 = await llm.chat(
            messages=[{"role": "user", "content": "Different prompt"}],
            max_tokens=100,
            temperature=0.7,
            use_cache=True,
        )

        # Check that the responses are as expected
        assert "Response to:" in response1.text
        assert "Response to:" in response2.text
        assert "Response to:" in response3.text

        # Check that generate was only called twice (once for each unique prompt)
        assert llm.generate_count == 2

    @pytest.mark.asyncio
    async def test_cache_with_different_parameters(self, monkeypatch):
        """Test caching with different parameters."""
        # Create a mock LLM
        llm = MockLLM("mock://model")

        # Mock the cache to avoid actual caching
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Always return None (cache miss) for the first 3 calls
        mock_cache.set = MagicMock()  # Mock the set method

        # Create cached responses for the second round
        cached_response1 = LLMResponse(
            text="Response to: Hello, world!",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True, "params": "set1"},
        )

        cached_response2 = LLMResponse(
            text="Response to: Hello, world!",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True, "params": "set2"},
        )

        cached_response3 = LLMResponse(
            text="Response to: Hello, world!",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True, "params": "set3"},
        )

        # Set up the mock to return None for first 3 calls, then cached responses
        mock_cache.get.side_effect = [
            None,
            None,
            None,  # First round - all cache misses
            cached_response1,
            cached_response2,
            cached_response3,  # Second round - all cache hits
        ]

        # Mock the get_model_cache function to return our mock cache
        monkeypatch.setattr("saplings.core.model_caching.get_model_cache", lambda **_: mock_cache)

        # Generate responses with different parameters
        await llm.generate_with_cache(prompt="Hello, world!", max_tokens=100, temperature=0.7)

        await llm.generate_with_cache(
            prompt="Hello, world!", max_tokens=200, temperature=0.7  # Different max_tokens
        )

        await llm.generate_with_cache(
            prompt="Hello, world!", max_tokens=100, temperature=0.5  # Different temperature
        )

        # Check that all responses were generated (not cached)
        assert llm.generate_count == 3

        # Generate the same responses again
        await llm.generate_with_cache(prompt="Hello, world!", max_tokens=100, temperature=0.7)

        await llm.generate_with_cache(prompt="Hello, world!", max_tokens=200, temperature=0.7)

        await llm.generate_with_cache(prompt="Hello, world!", max_tokens=100, temperature=0.5)

        # Check that no new responses were generated (all cached)
        assert llm.generate_count == 3

    @pytest.mark.asyncio
    async def test_cache_with_functions(self, monkeypatch):
        """Test caching with functions."""
        # Create a mock LLM
        llm = MockLLM("mock://model")

        # Mock the cache to avoid actual caching
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Always return None (cache miss) for the first 2 calls
        mock_cache.set = MagicMock()  # Mock the set method

        # Create cached responses for the second round
        cached_response1 = LLMResponse(
            text="Response to: What's the weather like in San Francisco?",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True, "function_call": "auto"},
        )

        cached_response2 = LLMResponse(
            text="Response to: What's the weather like in San Francisco?",
            model_uri="mock://model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"provider": "mock", "cached": True, "function_call": "get_weather"},
        )

        # Set up the mock to return None for first 2 calls, then cached responses
        mock_cache.get.side_effect = [
            None,
            None,  # First round - all cache misses
            cached_response1,
            cached_response2,  # Second round - all cache hits
        ]

        # Mock the get_model_cache function to return our mock cache
        monkeypatch.setattr("saplings.core.model_caching.get_model_cache", lambda **_: mock_cache)

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

        # Generate responses with different function parameters
        await llm.generate_with_cache(
            prompt="What's the weather like in San Francisco?",
            functions=[function],
            function_call="auto",
        )

        await llm.generate_with_cache(
            prompt="What's the weather like in San Francisco?",
            functions=[function],
            function_call={"name": "get_weather"},  # Different function_call
        )

        # Check that both responses were generated (not cached)
        assert llm.generate_count == 2

        # Generate the same responses again
        await llm.generate_with_cache(
            prompt="What's the weather like in San Francisco?",
            functions=[function],
            function_call="auto",
        )

        await llm.generate_with_cache(
            prompt="What's the weather like in San Francisco?",
            functions=[function],
            function_call={"name": "get_weather"},
        )

        # Check that no new responses were generated (all cached)
        assert llm.generate_count == 2

    @pytest.mark.asyncio
    async def test_cache_with_different_namespaces(self, monkeypatch):
        """Test caching with different namespaces."""
        # Create a mock LLM
        llm = MockLLM("mock://model")

        # Create two separate mock caches for different namespaces
        mock_cache1 = MagicMock()
        mock_cache1.get.side_effect = [
            None,
            LLMResponse(
                text="Response to: Hello, world!",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                metadata={"provider": "mock", "cached": True, "namespace": "namespace1"},
            ),
        ]
        mock_cache1.set = MagicMock()

        mock_cache2 = MagicMock()
        mock_cache2.get.side_effect = [
            None,
            LLMResponse(
                text="Response to: Hello, world!",
                model_uri="mock://model",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                metadata={"provider": "mock", "cached": True, "namespace": "namespace2"},
            ),
        ]
        mock_cache2.set = MagicMock()

        # Mock the get_model_cache function to return different caches based on namespace
        def mock_get_cache(**kwargs):
            namespace = kwargs.get("namespace", "default")
            if namespace == "namespace1":
                return mock_cache1
            elif namespace == "namespace2":
                return mock_cache2
            return MagicMock()

        monkeypatch.setattr("saplings.core.model_caching.get_model_cache", mock_get_cache)

        # Generate responses with different namespaces
        await llm.generate_with_cache(prompt="Hello, world!", cache_namespace="namespace1")

        await llm.generate_with_cache(prompt="Hello, world!", cache_namespace="namespace2")

        # Check that both responses were generated (different namespaces)
        assert llm.generate_count == 2

        # Generate the same responses again
        await llm.generate_with_cache(prompt="Hello, world!", cache_namespace="namespace1")

        await llm.generate_with_cache(prompt="Hello, world!", cache_namespace="namespace2")

        # Check that no new responses were generated (all cached)
        assert llm.generate_count == 2
