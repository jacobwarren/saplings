from __future__ import annotations

"""
Function authorization module for Saplings.

This module provides utilities for authorizing function calls.
"""


import functools
import logging
from enum import Enum

from saplings.core.function_registry import function_registry

logger = logging.getLogger(__name__)


class AuthorizationLevel(Enum):
    """Authorization levels for functions."""

    PUBLIC = 0
    USER = 1
    ADMIN = 2
    SYSTEM = 3


class FunctionAuthorizer:
    """Utility for authorizing function calls."""

    def __init__(self) -> None:
        """Initialize the function authorizer."""
        self._function_levels: dict[str, AuthorizationLevel] = {}
        self._function_groups: dict[str, AuthorizationLevel] = {}
        self._current_level = AuthorizationLevel.PUBLIC

    def set_function_level(self, name: str, level: AuthorizationLevel) -> None:
        """
        Set the authorization level for a function.

        Args:
        ----
            name: Name of the function
            level: Authorization level

        """
        self._function_levels[name] = level
        logger.debug(f"Set authorization level for function {name} to {level.name}")

    def set_group_level(self, group: str, level: AuthorizationLevel) -> None:
        """
        Set the authorization level for a function group.

        Args:
        ----
            group: Name of the group
            level: Authorization level

        """
        self._function_groups[group] = level
        logger.debug(f"Set authorization level for group {group} to {level.name}")

    def set_current_level(self, level: AuthorizationLevel) -> None:
        """
        Set the current authorization level.

        Args:
        ----
            level: Authorization level

        """
        self._current_level = level
        logger.debug(f"Set current authorization level to {level.name}")

    def get_function_level(self, name: str) -> AuthorizationLevel:
        """
        Get the authorization level for a function.

        Args:
        ----
            name: Name of the function

        Returns:
        -------
            AuthorizationLevel: Authorization level

        """
        # Check if the function has a specific level
        if name in self._function_levels:
            return self._function_levels[name]

        # Check if the function is in a group with a level
        func_info = function_registry.get_function(name)
        if func_info:
            for group in self._function_groups:
                if name in function_registry.get_group(group):
                    return self._function_groups[group]

        # Default to PUBLIC
        return AuthorizationLevel.PUBLIC

    def is_authorized(self, name: str) -> bool:
        """
        Check if the current level is authorized to call a function.

        Args:
        ----
            name: Name of the function

        Returns:
        -------
            bool: True if authorized, False otherwise

        """
        function_level = self.get_function_level(name)
        return self._current_level.value >= function_level.value

    def authorize_function_call(self, name: str) -> None:
        """
        Authorize a function call.

        Args:
        ----
            name: Name of the function

        Raises:
        ------
            PermissionError: If not authorized

        """
        if not self.is_authorized(name):
            function_level = self.get_function_level(name)
            msg = (
                f"Not authorized to call function {name}. "
                f"Required level: {function_level.name}, "
                f"Current level: {self._current_level.name}"
            )
            raise PermissionError(msg)

    def get_authorized_functions(self):
        """
        Get a list of authorized functions.

        Returns
        -------
            List[str]: List of authorized function names

        """
        authorized = []

        # Check all registered functions
        for name in function_registry.get_all_functions():
            if self.is_authorized(name):
                authorized.append(name)

        return authorized

    def get_authorized_groups(self):
        """
        Get a list of authorized function groups.

        Returns
        -------
            List[str]: List of authorized group names

        """
        authorized = []

        # Check all groups
        for group, level in self._function_groups.items():
            if self._current_level.value >= level.value:
                authorized.append(group)

        return authorized


# Create a singleton instance
function_authorizer = FunctionAuthorizer()


def requires_level(level: AuthorizationLevel):
    """
    Decorator for requiring an authorization level.

    Args:
    ----
        level: Required authorization level

    Returns:
    -------
        Callable: Decorated function

    """

    def decorator(func):
        # Register the function with the authorizer
        function_authorizer.set_function_level(func.__name__, level)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check authorization
            function_authorizer.authorize_function_call(func.__name__)

            # Call the function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def set_current_level(level: AuthorizationLevel) -> None:
    """
    Set the current authorization level.

    Args:
    ----
        level: Authorization level

    """
    function_authorizer.set_current_level(level)


def set_function_level(name: str, level: AuthorizationLevel) -> None:
    """
    Set the authorization level for a function.

    Args:
    ----
        name: Name of the function
        level: Authorization level

    """
    function_authorizer.set_function_level(name, level)


def set_group_level(group: str, level: AuthorizationLevel) -> None:
    """
    Set the authorization level for a function group.

    Args:
    ----
        group: Name of the group
        level: Authorization level

    """
    function_authorizer.set_group_level(group, level)


def get_authorized_functions():
    """
    Get a list of authorized functions.

    Returns
    -------
        List[str]: List of authorized function names

    """
    return function_authorizer.get_authorized_functions()


def get_authorized_groups():
    """
    Get a list of authorized function groups.

    Returns
    -------
        List[str]: List of authorized group names

    """
    return function_authorizer.get_authorized_groups()
