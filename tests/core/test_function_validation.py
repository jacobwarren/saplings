"""
Tests for function validation.

This module provides tests for the function validation utilities in Saplings.
"""

import json
from typing import Dict, List, Optional

import pytest
from pydantic import ValidationError

from saplings.core.function_registry import FunctionRegistry, function_registry
from saplings.core.function_validation import FunctionValidator, validate_function_call


class TestFunctionValidator:
    """Test class for function validation."""

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
    def validator(self):
        """Create a function validator for testing."""
        return FunctionValidator()

    def test_validate_function_call(self, registry, validator):
        """Test validating a function call."""
        # Define a test function
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Register the function
        registry.register(add)

        # Validate valid arguments
        validated = validator.validate_function_call("add", {"a": 1, "b": 2})
        assert validated == {"a": 1, "b": 2}

        # Validate arguments with type conversion
        validated = validator.validate_function_call("add", {"a": "1", "b": "2"})
        assert validated == {"a": 1, "b": 2}

        # Validate invalid arguments
        with pytest.raises(ValidationError):
            validator.validate_function_call("add", {"a": "not_a_number", "b": 2})

        # Validate missing required arguments
        with pytest.raises(ValidationError):
            validator.validate_function_call("add", {"a": 1})

    def test_validate_function_call_with_default_values(self, registry, validator):
        """Test validating a function call with default values."""
        # Define a test function with default values
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        # Register the function
        registry.register(greet)

        # Validate with all arguments
        validated = validator.validate_function_call("greet", {"name": "World", "greeting": "Hi"})
        assert validated == {"name": "World", "greeting": "Hi"}

        # Validate with only required arguments
        validated = validator.validate_function_call("greet", {"name": "World"})
        assert validated == {"name": "World", "greeting": "Hello"}

    def test_validate_function_call_with_complex_types(self, registry, validator):
        """Test validating a function call with complex types."""
        # Define a test function with complex types
        def process_data(items: str, options: str = None) -> Dict:
            """Process data.

            Args:
                items: JSON string of items
                options: JSON string of options
            """
            items_list = json.loads(items)
            options_dict = json.loads(options) if options else {}
            return {"items": items_list, "options": options_dict}

        # Register the function
        registry.register(process_data)

        # Validate with valid arguments
        validated = validator.validate_function_call(
            "process_data",
            {"items": json.dumps([1, 2, 3]), "options": json.dumps({"mode": "fast"})}
        )
        assert validated == {"items": json.dumps([1, 2, 3]), "options": json.dumps({"mode": "fast"})}

        # Validate with type conversion
        validated = validator.validate_function_call(
            "process_data",
            {"items": json.dumps(["1", "2", "3"])}
        )
        assert validated == {"items": json.dumps(["1", "2", "3"]), "options": None}

        # Validate with invalid arguments
        with pytest.raises(ValidationError):
            validator.validate_function_call(
                "process_data",
                {"items": 123}  # Not a string
            )

    def test_validate_function_call_with_json_string(self, registry, validator):
        """Test validating a function call with a JSON string."""
        # Define a test function
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Register the function
        registry.register(add)

        # Validate with a JSON string
        validated = validator.validate_function_call("add", json.dumps({"a": 1, "b": 2}))
        assert validated == {"a": 1, "b": 2}

    def test_validate_nonexistent_function(self, validator):
        """Test validating a call to a nonexistent function."""
        with pytest.raises(ValueError, match="Function not registered: nonexistent"):
            validator.validate_function_call("nonexistent", {})

    def test_validate_function_call_convenience(self, registry):
        """Test the convenience function for validating function calls."""
        # Define a test function
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Register the function
        registry.register(add)

        # Validate valid arguments
        validated = validate_function_call("add", {"a": 1, "b": 2})
        assert validated == {"a": 1, "b": 2}
