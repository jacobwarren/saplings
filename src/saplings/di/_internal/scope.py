from __future__ import annotations

"""
Scope module for dependency injection.

This module provides scoped lifetimes for services in the dependency injection system.
"""

import logging
import threading
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Type

from saplings.di._internal.exceptions import ScopeError

# Configure logging
logger = logging.getLogger(__name__)


class LifecycleScope(Enum):
    """Lifecycle scopes for services."""

    SINGLETON = auto()
    """Service is created once per container."""

    SCOPED = auto()
    """Service is created once per scope."""

    TRANSIENT = auto()
    """Service is created each time it is resolved."""


class Scope:
    """
    Scope for dependency injection.

    A scope manages the lifetime of services and provides a context
    for resolving scoped services.
    """

    def __init__(self, parent: Optional[Scope] = None, name: str = "default"):
        """
        Initialize the scope.

        Args:
        ----
            parent: The parent scope, if any
            name: The name of the scope

        """
        self.parent = parent
        self.name = name
        self.instances: Dict[Type, Any] = {}
        self.factories: Dict[Type, Callable[[], Any]] = {}
        self.lock = threading.RLock()
        logger.debug(f"Created scope '{name}'")

    def register(self, service_type: Type, factory: Callable[[], Any]) -> None:
        """
        Register a factory for a service type.

        Args:
        ----
            service_type: The type to register
            factory: The factory function to create instances

        """
        with self.lock:
            self.factories[service_type] = factory
            logger.debug(f"Registered factory for '{service_type.__name__}' in scope '{self.name}'")

    def resolve(self, service_type: Type) -> Any:
        """
        Resolve a service from the scope.

        Args:
        ----
            service_type: The type to resolve

        Returns:
        -------
            An instance of the requested type

        Raises:
        ------
            ScopeError: If the service cannot be resolved

        """
        # Check if we have an instance
        with self.lock:
            if service_type in self.instances:
                return self.instances[service_type]

            # Check if we have a factory
            if service_type in self.factories:
                instance = self.factories[service_type]()
                self.instances[service_type] = instance
                return instance

            # Check parent scope
            if self.parent:
                return self.parent.resolve(service_type)

            # Service not found
            raise ScopeError(f"Service '{service_type.__name__}' not found in scope '{self.name}'")

    def dispose(self) -> None:
        """
        Dispose of the scope and all its instances.

        This method disposes of all instances in the scope that implement
        a `dispose` or `close` method.
        """
        with self.lock:
            # Dispose of all instances
            for service_type, instance in self.instances.items():
                try:
                    # Try to dispose of the instance
                    if hasattr(instance, "dispose") and callable(instance.dispose):
                        instance.dispose()
                    elif hasattr(instance, "close") and callable(instance.close):
                        instance.close()
                except Exception as e:
                    logger.warning(f"Error disposing instance of '{service_type.__name__}': {e}")

            # Clear instances and factories
            self.instances.clear()
            self.factories.clear()
            logger.debug(f"Disposed scope '{self.name}'")


class ScopeManager:
    """
    Manager for dependency injection scopes.

    This class manages the creation and disposal of scopes for the
    dependency injection system.
    """

    def __init__(self):
        """Initialize the scope manager."""
        self.root_scope = Scope(name="root")
        self.current_scope = self.root_scope
        self.scopes = {self.root_scope.name: self.root_scope}
        self.lock = threading.RLock()
        self._scope_stack = []
        logger.debug("Initialized scope manager")

    @contextmanager
    def create_scope(self, name: Optional[str] = None) -> Scope:
        """
        Create a new scope.

        Args:
        ----
            name: The name of the scope

        Returns:
        -------
            The created scope

        Raises:
        ------
            ScopeError: If a scope with the given name already exists

        """
        # Generate a unique name if none provided
        if name is None:
            name = f"scope_{len(self.scopes)}"

        # Check if the name is already used
        with self.lock:
            if name in self.scopes:
                raise ScopeError(f"Scope with name '{name}' already exists")

            # Create the scope
            scope = Scope(parent=self.current_scope, name=name)
            self.scopes[name] = scope

            # Save the previous scope
            previous_scope = self.current_scope
            self.current_scope = scope
            self._scope_stack.append(previous_scope)

            try:
                # Yield the scope
                yield scope
            finally:
                # Restore the previous scope
                self.current_scope = self._scope_stack.pop()

                # Dispose of the scope
                scope.dispose()

                # Remove the scope
                with self.lock:
                    if name in self.scopes:
                        del self.scopes[name]

    def get_current_scope(self) -> Scope:
        """
        Get the current scope.

        Returns
        -------
            The current scope

        """
        return self.current_scope

    def reset(self) -> None:
        """
        Reset the scope manager.

        This method disposes of all scopes and creates a new root scope.
        """
        with self.lock:
            # Dispose of all scopes
            for scope in self.scopes.values():
                scope.dispose()

            # Create a new root scope
            self.root_scope = Scope(name="root")
            self.current_scope = self.root_scope
            self.scopes = {self.root_scope.name: self.root_scope}
            self._scope_stack = []
            logger.debug("Reset scope manager")
