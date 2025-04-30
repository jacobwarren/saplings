"""
Tests for the function registry.

This module provides tests for the function registry in Saplings.
"""

import pytest
from typing import Dict, List, Optional

from saplings.core.function_registry import FunctionRegistry, function_registry, register_function


class TestFunctionRegistry:
    """Test class for the function registry."""
    
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
    
    def test_register_function(self, registry):
        """Test registering a function."""
        # Define a test function
        def test_func(param1: str, param2: int = 0) -> str:
            """Test function.
            
            Args:
                param1: First parameter
                param2: Second parameter
            
            Returns:
                str: Result
            """
            return f"{param1} {param2}"
        
        # Register the function
        registry.register(test_func)
        
        # Check that the function was registered
        assert "test_func" in registry.get_all_functions()
        
        # Check function info
        func_info = registry.get_function("test_func")
        assert func_info["name"] == "test_func"
        assert func_info["description"] == "Test function."
        assert "param1" in func_info["parameters"]
        assert "param2" in func_info["parameters"]
        assert func_info["parameters"]["param1"]["type"] == "string"
        assert func_info["parameters"]["param2"]["type"] == "integer"
        assert func_info["required"] == ["param1"]
    
    def test_register_function_with_decorator(self, registry):
        """Test registering a function with a decorator."""
        # Define a test function with a decorator
        @registry.register(description="Custom description")
        def test_func(param1: str, param2: int = 0) -> str:
            return f"{param1} {param2}"
        
        # Check that the function was registered
        assert "test_func" in registry.get_all_functions()
        
        # Check function info
        func_info = registry.get_function("test_func")
        assert func_info["name"] == "test_func"
        assert func_info["description"] == "Custom description"
    
    def test_register_function_with_group(self, registry):
        """Test registering a function with a group."""
        # Define test functions
        def test_func1(param: str) -> str:
            return param
        
        def test_func2(param: int) -> int:
            return param
        
        # Register the functions with groups
        registry.register(test_func1, group="group1")
        registry.register(test_func2, group="group2")
        
        # Check that the functions were registered
        assert "test_func1" in registry.get_all_functions()
        assert "test_func2" in registry.get_all_functions()
        
        # Check groups
        assert "test_func1" in registry.get_group("group1")
        assert "test_func2" in registry.get_group("group2")
        assert "test_func1" not in registry.get_group("group2")
        assert "test_func2" not in registry.get_group("group1")
    
    def test_get_function_definition(self, registry):
        """Test getting a function definition."""
        # Define a test function
        def get_weather(location: str, unit: str = "celsius") -> Dict:
            """Get the weather for a location.
            
            Args:
                location: The location to get weather for
                unit: The unit to use (celsius or fahrenheit)
            
            Returns:
                Dict: Weather information
            """
            return {"location": location, "unit": unit}
        
        # Register the function
        registry.register(get_weather)
        
        # Get the function definition
        func_def = registry.get_function_definition("get_weather")
        
        # Check the definition
        assert func_def["name"] == "get_weather"
        assert func_def["description"] == "Get the weather for a location."
        assert func_def["parameters"]["type"] == "object"
        assert "location" in func_def["parameters"]["properties"]
        assert "unit" in func_def["parameters"]["properties"]
        assert func_def["parameters"]["required"] == ["location"]
    
    def test_get_group_definitions(self, registry):
        """Test getting function definitions for a group."""
        # Define test functions
        def test_func1(param1: str) -> str:
            """Function 1."""
            return param1
        
        def test_func2(param2: int) -> int:
            """Function 2."""
            return param2
        
        # Register the functions with the same group
        registry.register(test_func1, group="test_group")
        registry.register(test_func2, group="test_group")
        
        # Get the group definitions
        group_defs = registry.get_group_definitions("test_group")
        
        # Check the definitions
        assert len(group_defs) == 2
        assert any(d["name"] == "test_func1" for d in group_defs)
        assert any(d["name"] == "test_func2" for d in group_defs)
    
    def test_call_function(self, registry):
        """Test calling a registered function."""
        # Define a test function
        def add(a: int, b: int) -> int:
            """Add two numbers.
            
            Args:
                a: First number
                b: Second number
            
            Returns:
                int: Sum of the numbers
            """
            return a + b
        
        # Register the function
        registry.register(add)
        
        # Call the function
        result = registry.call_function("add", {"a": 1, "b": 2})
        
        # Check the result
        assert result == 3
    
    def test_unregister_function(self, registry):
        """Test unregistering a function."""
        # Define a test function
        def test_func() -> None:
            pass
        
        # Register the function
        registry.register(test_func, group="test_group")
        
        # Check that the function was registered
        assert "test_func" in registry.get_all_functions()
        assert "test_func" in registry.get_group("test_group")
        
        # Unregister the function
        result = registry.unregister("test_func")
        
        # Check that the function was unregistered
        assert result is True
        assert "test_func" not in registry.get_all_functions()
        assert "test_func" not in registry.get_group("test_group")
    
    def test_singleton(self):
        """Test that FunctionRegistry is a singleton."""
        registry1 = FunctionRegistry()
        registry2 = FunctionRegistry()
        
        # Check that they are the same instance
        assert registry1 is registry2
    
    def test_global_registry(self):
        """Test the global function registry."""
        # Define a test function
        @register_function(group="global_test")
        def global_test_func(param: str) -> str:
            """Global test function."""
            return param
        
        # Check that the function was registered
        assert "global_test_func" in function_registry.get_all_functions()
        assert "global_test_func" in function_registry.get_group("global_test")
        
        # Clean up
        function_registry.unregister("global_test_func")
