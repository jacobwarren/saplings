from __future__ import annotations

"""
Exceptions for the dependency injection system.

This module defines exceptions specific to the dependency injection system.
"""


class DIError(Exception):
    """Base class for all dependency injection errors."""


class ServiceNotRegisteredError(DIError):
    """Raised when a service is not registered with the container."""

    def __init__(self, service_type):
        """
        Initialize the error.

        Args:
        ----
            service_type: The type that was not registered

        """
        self.service_type = service_type
        type_name = getattr(service_type, "__name__", str(service_type))
        super().__init__(f"Service '{type_name}' is not registered with the container")


class CircularDependencyError(DIError):
    """Raised when a circular dependency is detected."""

    def __init__(self, dependency_chain):
        """
        Initialize the error.

        Args:
        ----
            dependency_chain: The chain of dependencies that form a cycle

        """
        self.dependency_chain = dependency_chain
        chain_str = " -> ".join(getattr(t, "__name__", str(t)) for t in dependency_chain)
        super().__init__(f"Circular dependency detected: {chain_str}")


class ResolutionError(DIError):
    """Raised when a service cannot be resolved."""

    def __init__(self, service_type, cause=None):
        """
        Initialize the error.

        Args:
        ----
            service_type: The type that could not be resolved
            cause: The underlying cause of the error

        """
        self.service_type = service_type
        self.cause = cause
        type_name = getattr(service_type, "__name__", str(service_type))
        message = f"Failed to resolve service '{type_name}'"
        if cause:
            message += f": {cause!s}"
        super().__init__(message)


class ScopeError(DIError):
    """Raised when there is an error with a scope."""

    def __init__(self, message):
        """
        Initialize the error.

        Args:
        ----
            message: The error message

        """
        super().__init__(message)


class InitializationError(DIError):
    """Raised when a service fails to initialize."""

    def __init__(self, service_type, cause=None):
        """
        Initialize the error.

        Args:
        ----
            service_type: The type that failed to initialize
            cause: The underlying cause of the error

        """
        self.service_type = service_type
        self.cause = cause
        type_name = getattr(service_type, "__name__", str(service_type))
        message = f"Failed to initialize service '{type_name}'"
        if cause:
            message += f": {cause!s}"
        super().__init__(message)
