"""
Tests for parallel function calling.

This module provides tests for the parallel function calling in Saplings.
"""

import asyncio
import time
from typing import Dict, List

import pytest

from saplings.core.function_registry import FunctionRegistry, function_registry
from saplings.core.parallel_function import ParallelFunctionCaller, call_functions_parallel


class TestParallelFunctionCaller:
    """Test class for parallel function calling."""
    
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
        """Create a parallel function caller for testing."""
        return ParallelFunctionCaller()
    
    @pytest.mark.asyncio
    async def test_call_function(self, registry, caller):
        """Test calling a single function."""
        # Define a test function
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        # Register the function
        registry.register(add)
        
        # Call the function
        result = await caller.call_function("add", {"a": 1, "b": 2})
        
        # Check the result
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_call_functions_parallel(self, registry, caller):
        """Test calling multiple functions in parallel."""
        # Define test functions
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b
        
        def slow_function(delay: float) -> float:
            """A slow function that sleeps."""
            time.sleep(delay)
            return delay
        
        # Register the functions
        registry.register(add)
        registry.register(multiply)
        registry.register(slow_function)
        
        # Call the functions in parallel
        function_calls = [
            {"name": "add", "arguments": {"a": 1, "b": 2}},
            {"name": "multiply", "arguments": {"a": 3, "b": 4}},
            {"name": "slow_function", "arguments": {"delay": 0.1}},
        ]
        
        results = await caller.call_functions(function_calls)
        
        # Check the results
        assert len(results) == 3
        assert results[0] == ("add", 3)
        assert results[1] == ("multiply", 12)
        assert results[2][0] == "slow_function"
        assert abs(results[2][1] - 0.1) < 0.05  # Allow for small timing differences
    
    @pytest.mark.asyncio
    async def test_call_functions_with_timeout(self, registry, caller):
        """Test calling functions with a timeout."""
        # Define a slow function
        def slow_function(delay: float) -> float:
            """A slow function that sleeps."""
            time.sleep(delay)
            return delay
        
        # Register the function
        registry.register(slow_function)
        
        # Call the function with a timeout
        with pytest.raises(asyncio.TimeoutError):
            await caller.call_function("slow_function", {"delay": 0.5}, timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_call_async_function(self, registry, caller):
        """Test calling an async function."""
        # Define an async function
        async def async_add(a: int, b: int) -> int:
            """Add two numbers asynchronously."""
            await asyncio.sleep(0.1)
            return a + b
        
        # Register the function
        registry.register(async_add)
        
        # Call the function
        result = await caller.call_function("async_add", {"a": 1, "b": 2})
        
        # Check the result
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_call_functions_parallel_convenience(self, registry):
        """Test the convenience function for parallel function calling."""
        # Define test functions
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b
        
        # Register the functions
        registry.register(add)
        registry.register(multiply)
        
        # Call the functions in parallel
        function_calls = [
            {"name": "add", "arguments": {"a": 1, "b": 2}},
            {"name": "multiply", "arguments": {"a": 3, "b": 4}},
        ]
        
        results = await call_functions_parallel(function_calls)
        
        # Check the results
        assert len(results) == 2
        assert results[0] == ("add", 3)
        assert results[1] == ("multiply", 12)
    
    @pytest.mark.asyncio
    async def test_call_functions_with_string_arguments(self, registry, caller):
        """Test calling functions with string arguments."""
        # Define a test function
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        # Register the function
        registry.register(add)
        
        # Call the function with string arguments
        function_calls = [
            {"name": "add", "arguments": '{"a": 1, "b": 2}'},
        ]
        
        results = await caller.call_functions(function_calls)
        
        # Check the results
        assert len(results) == 1
        assert results[0] == ("add", 3)
    
    @pytest.mark.asyncio
    async def test_call_nonexistent_function(self, caller):
        """Test calling a function that doesn't exist."""
        # Call a function that doesn't exist
        with pytest.raises(ValueError, match="Function not registered: nonexistent"):
            await caller.call_function("nonexistent", {})
    
    @pytest.mark.asyncio
    async def test_call_functions_with_exceptions(self, registry, caller):
        """Test calling functions that raise exceptions."""
        # Define a function that raises an exception
        def failing_function() -> None:
            """A function that raises an exception."""
            raise ValueError("Test error")
        
        # Define a normal function
        def normal_function() -> str:
            """A normal function."""
            return "success"
        
        # Register the functions
        registry.register(failing_function)
        registry.register(normal_function)
        
        # Call the functions in parallel
        function_calls = [
            {"name": "failing_function", "arguments": {}},
            {"name": "normal_function", "arguments": {}},
        ]
        
        results = await caller.call_functions(function_calls)
        
        # Check the results
        assert len(results) == 2
        assert results[0][0] == "failing_function"
        assert isinstance(results[0][1], ValueError)
        assert results[1] == ("normal_function", "success")
