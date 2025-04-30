"""
Tests for function authorization.

This module provides tests for the function authorization utilities in Saplings.
"""

import pytest

from saplings.core.function_authorization import (
    AuthorizationLevel,
    FunctionAuthorizer,
    get_authorized_functions,
    get_authorized_groups,
    requires_level,
    set_current_level,
    set_function_level,
    set_group_level,
)
from saplings.core.function_registry import FunctionRegistry, function_registry


class TestFunctionAuthorizer:
    """Test class for function authorization."""
    
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
    def authorizer(self):
        """Create a function authorizer for testing."""
        authorizer = FunctionAuthorizer()
        
        # Save the original state
        original_levels = authorizer._function_levels.copy()
        original_groups = authorizer._function_groups.copy()
        original_current = authorizer._current_level
        
        # Clear the state
        authorizer._function_levels.clear()
        authorizer._function_groups.clear()
        authorizer._current_level = AuthorizationLevel.PUBLIC
        
        yield authorizer
        
        # Restore the original state
        authorizer._function_levels = original_levels
        authorizer._function_groups = original_groups
        authorizer._current_level = original_current
    
    def test_set_function_level(self, authorizer):
        """Test setting the authorization level for a function."""
        # Set the level
        authorizer.set_function_level("func1", AuthorizationLevel.ADMIN)
        
        # Check the level
        assert authorizer.get_function_level("func1") == AuthorizationLevel.ADMIN
        
        # Check a function without a specific level
        assert authorizer.get_function_level("func2") == AuthorizationLevel.PUBLIC
    
    def test_set_group_level(self, registry, authorizer):
        """Test setting the authorization level for a function group."""
        # Define test functions
        def func1():
            pass
        
        def func2():
            pass
        
        # Register the functions with groups
        registry.register(func1, group="group1")
        registry.register(func2, group="group2")
        
        # Set the group level
        authorizer.set_group_level("group1", AuthorizationLevel.ADMIN)
        
        # Check the levels
        assert authorizer.get_function_level("func1") == AuthorizationLevel.ADMIN
        assert authorizer.get_function_level("func2") == AuthorizationLevel.PUBLIC
    
    def test_set_current_level(self, authorizer):
        """Test setting the current authorization level."""
        # Set the current level
        authorizer.set_current_level(AuthorizationLevel.ADMIN)
        
        # Check the current level
        assert authorizer._current_level == AuthorizationLevel.ADMIN
    
    def test_is_authorized(self, authorizer):
        """Test checking if a function is authorized."""
        # Set function levels
        authorizer.set_function_level("public_func", AuthorizationLevel.PUBLIC)
        authorizer.set_function_level("user_func", AuthorizationLevel.USER)
        authorizer.set_function_level("admin_func", AuthorizationLevel.ADMIN)
        
        # Set current level to PUBLIC
        authorizer.set_current_level(AuthorizationLevel.PUBLIC)
        
        # Check authorization
        assert authorizer.is_authorized("public_func") is True
        assert authorizer.is_authorized("user_func") is False
        assert authorizer.is_authorized("admin_func") is False
        
        # Set current level to USER
        authorizer.set_current_level(AuthorizationLevel.USER)
        
        # Check authorization
        assert authorizer.is_authorized("public_func") is True
        assert authorizer.is_authorized("user_func") is True
        assert authorizer.is_authorized("admin_func") is False
        
        # Set current level to ADMIN
        authorizer.set_current_level(AuthorizationLevel.ADMIN)
        
        # Check authorization
        assert authorizer.is_authorized("public_func") is True
        assert authorizer.is_authorized("user_func") is True
        assert authorizer.is_authorized("admin_func") is True
    
    def test_authorize_function_call(self, authorizer):
        """Test authorizing a function call."""
        # Set function levels
        authorizer.set_function_level("public_func", AuthorizationLevel.PUBLIC)
        authorizer.set_function_level("admin_func", AuthorizationLevel.ADMIN)
        
        # Set current level to PUBLIC
        authorizer.set_current_level(AuthorizationLevel.PUBLIC)
        
        # Authorize a public function
        authorizer.authorize_function_call("public_func")  # Should not raise
        
        # Authorize an admin function
        with pytest.raises(PermissionError):
            authorizer.authorize_function_call("admin_func")
        
        # Set current level to ADMIN
        authorizer.set_current_level(AuthorizationLevel.ADMIN)
        
        # Authorize an admin function
        authorizer.authorize_function_call("admin_func")  # Should not raise
    
    def test_get_authorized_functions(self, registry, authorizer):
        """Test getting a list of authorized functions."""
        # Define test functions
        def public_func():
            pass
        
        def user_func():
            pass
        
        def admin_func():
            pass
        
        # Register the functions
        registry.register(public_func)
        registry.register(user_func)
        registry.register(admin_func)
        
        # Set function levels
        authorizer.set_function_level("public_func", AuthorizationLevel.PUBLIC)
        authorizer.set_function_level("user_func", AuthorizationLevel.USER)
        authorizer.set_function_level("admin_func", AuthorizationLevel.ADMIN)
        
        # Set current level to PUBLIC
        authorizer.set_current_level(AuthorizationLevel.PUBLIC)
        
        # Get authorized functions
        authorized = authorizer.get_authorized_functions()
        assert "public_func" in authorized
        assert "user_func" not in authorized
        assert "admin_func" not in authorized
        
        # Set current level to ADMIN
        authorizer.set_current_level(AuthorizationLevel.ADMIN)
        
        # Get authorized functions
        authorized = authorizer.get_authorized_functions()
        assert "public_func" in authorized
        assert "user_func" in authorized
        assert "admin_func" in authorized
    
    def test_get_authorized_groups(self, authorizer):
        """Test getting a list of authorized function groups."""
        # Set group levels
        authorizer.set_group_level("public_group", AuthorizationLevel.PUBLIC)
        authorizer.set_group_level("user_group", AuthorizationLevel.USER)
        authorizer.set_group_level("admin_group", AuthorizationLevel.ADMIN)
        
        # Set current level to PUBLIC
        authorizer.set_current_level(AuthorizationLevel.PUBLIC)
        
        # Get authorized groups
        authorized = authorizer.get_authorized_groups()
        assert "public_group" in authorized
        assert "user_group" not in authorized
        assert "admin_group" not in authorized
        
        # Set current level to ADMIN
        authorizer.set_current_level(AuthorizationLevel.ADMIN)
        
        # Get authorized groups
        authorized = authorizer.get_authorized_groups()
        assert "public_group" in authorized
        assert "user_group" in authorized
        assert "admin_group" in authorized
    
    def test_requires_level_decorator(self, registry):
        """Test the requires_level decorator."""
        # Define a function with the decorator
        @requires_level(AuthorizationLevel.ADMIN)
        def admin_func():
            return "admin result"
        
        # Register the function
        registry.register(admin_func)
        
        # Set current level to USER
        set_current_level(AuthorizationLevel.USER)
        
        # Call the function
        with pytest.raises(PermissionError):
            admin_func()
        
        # Set current level to ADMIN
        set_current_level(AuthorizationLevel.ADMIN)
        
        # Call the function
        result = admin_func()
        assert result == "admin result"
    
    def test_convenience_functions(self, registry):
        """Test the convenience functions."""
        # Define test functions
        def public_func():
            pass
        
        def user_func():
            pass
        
        # Register the functions
        registry.register(public_func, group="public_group")
        registry.register(user_func, group="user_group")
        
        # Set levels
        set_function_level("public_func", AuthorizationLevel.PUBLIC)
        set_function_level("user_func", AuthorizationLevel.USER)
        set_group_level("public_group", AuthorizationLevel.PUBLIC)
        set_group_level("user_group", AuthorizationLevel.USER)
        
        # Set current level to PUBLIC
        set_current_level(AuthorizationLevel.PUBLIC)
        
        # Get authorized functions and groups
        authorized_funcs = get_authorized_functions()
        authorized_groups = get_authorized_groups()
        
        assert "public_func" in authorized_funcs
        assert "user_func" not in authorized_funcs
        assert "public_group" in authorized_groups
        assert "user_group" not in authorized_groups
