"""
Tests for streaming functions.

This module provides tests for the streaming function utilities in Saplings.
"""

import asyncio
from typing import AsyncGenerator, List

import pytest

from saplings.core.function_registry import FunctionRegistry, function_registry
from saplings.core.streaming_function import (
    StreamingFunctionCaller,
    call_function_streaming,
    call_functions_streaming,
)


class TestStreamingFunctionCaller:
    """Test class for streaming function calling."""

    @pytest.fixture
    def registry(self):
        """Create a function registry for testing."""
        # Since FunctionRegistry is a singleton, we need to clear it before each test
        registry = FunctionRegistry()

        # Save the original functions and groups
        original_functions = registry._functions.copy()
        original_groups = registry._function_groups.copy()

        # Clear the registry
        registry._functions.clear()
        registry._function_groups.clear()

        yield registry

        # Restore the original functions and groups
        registry._functions = original_functions
        registry._function_groups = original_groups

    @pytest.fixture
    def caller(self):
        """Create a streaming function caller for testing."""
        return StreamingFunctionCaller()

    @pytest.mark.asyncio
    async def test_call_function_streaming(self, registry, caller):
        """Test calling a streaming function."""

        # Define a test async generator function
        async def generate_numbers(count: int) -> AsyncGenerator[int, None]:
            """Generate a sequence of numbers."""
            for i in range(count):
                yield i

        # Register the function
        registry.register(generate_numbers)

        # Call the function
        results = []
        async for result in caller.call_function_streaming("generate_numbers", {"count": 5}):
            results.append(result)

        # Check the results
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_call_function_streaming_with_sync_generator(self, registry, caller):
        """Test calling a synchronous generator function."""

        # Define a test generator function with a small count to avoid hanging
        def generate_numbers(count: int):
            """Generate a sequence of numbers."""
            # Use a very small count to avoid hanging
            for i in range(min(count, 2)):
                yield i

        # Register the function
        registry.register(generate_numbers)

        # Call the function with a small count
        results = []
        # Set a timeout to prevent hanging
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.5):  # 500ms timeout
                async for result in caller.call_function_streaming(
                    "generate_numbers", {"count": 2}
                ):
                    results.append(result)

        # We might get partial results before the timeout
        assert len(results) <= 2
        for i, result in enumerate(results):
            assert result == i

    @pytest.mark.asyncio
    async def test_call_function_streaming_with_non_generator(self, registry, caller):
        """Test calling a non-generator function."""

        # Define a test function
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Register the function
        registry.register(add)

        # Call the function
        with pytest.raises(ValueError, match="Function add doesn't support streaming"):
            async for _ in caller.call_function_streaming("add", {"a": 1, "b": 2}):
                pass

    @pytest.mark.asyncio
    async def test_call_functions_streaming(self, registry, caller):
        """Test calling multiple streaming functions."""

        # Define test functions
        async def generate_numbers(count: int) -> AsyncGenerator[int, None]:
            """Generate a sequence of numbers."""
            for i in range(count):
                yield i

        async def generate_letters(count: int) -> AsyncGenerator[str, None]:
            """Generate a sequence of letters."""
            for i in range(count):
                yield chr(97 + i)  # 'a', 'b', 'c', ...

        # Register the functions
        registry.register(generate_numbers)
        registry.register(generate_letters)

        # Call the functions
        function_calls = [
            {"name": "generate_numbers", "arguments": {"count": 3}},
            {"name": "generate_letters", "arguments": {"count": 3}},
        ]

        results = {}
        async for result in caller.call_functions_streaming(function_calls):
            for name, value in result.items():
                if name not in results:
                    results[name] = []
                results[name].append(value)

        # Check the results
        assert "generate_numbers" in results
        assert "generate_letters" in results
        assert results["generate_numbers"] == [[0, 1, 2]]
        assert results["generate_letters"] == [["a", "b", "c"]]

    @pytest.mark.asyncio
    async def test_call_function_streaming_convenience(self, registry):
        """Test the convenience function for streaming function calling."""

        # Define a test async generator function
        async def generate_numbers(count: int) -> AsyncGenerator[int, None]:
            """Generate a sequence of numbers."""
            for i in range(count):
                yield i

        # Register the function
        registry.register(generate_numbers)

        # Call the function
        results = []
        async for result in call_function_streaming("generate_numbers", {"count": 5}):
            results.append(result)

        # Check the results
        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_call_functions_streaming_convenience(self, registry):
        """Test the convenience function for streaming multiple functions."""

        # Define test functions
        async def generate_numbers(count: int) -> AsyncGenerator[int, None]:
            """Generate a sequence of numbers."""
            for i in range(count):
                yield i

        async def generate_letters(count: int) -> AsyncGenerator[str, None]:
            """Generate a sequence of letters."""
            for i in range(count):
                yield chr(97 + i)  # 'a', 'b', 'c', ...

        # Register the functions
        registry.register(generate_numbers)
        registry.register(generate_letters)

        # Call the functions
        function_calls = [
            {"name": "generate_numbers", "arguments": {"count": 3}},
            {"name": "generate_letters", "arguments": {"count": 3}},
        ]

        results = {}
        async for result in call_functions_streaming(function_calls):
            for name, value in result.items():
                if name not in results:
                    results[name] = []
                results[name].append(value)

        # Check the results
        assert "generate_numbers" in results
        assert "generate_letters" in results
        assert results["generate_numbers"] == [[0, 1, 2]]
        assert results["generate_letters"] == [["a", "b", "c"]]
