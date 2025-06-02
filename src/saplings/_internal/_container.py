from __future__ import annotations

"""
Container module for dependency injection in Saplings.

This module provides a dependency injection container that manages
the lifecycle and dependencies of all services used in the framework.
It uses the dependency-injector library to implement IoC (Inversion of Control)
principles and provides:

- Multiple lifecycle scopes (singleton, transient, scoped)
- Circular dependency detection
- Lazy singleton initialization
- Instance tracking for memory management
"""

import enum
import logging
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, cast

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic type hints
T = TypeVar("T")
U = TypeVar("U")


class CircularDependencyError(Exception):
    """Exception raised when a circular dependency is detected."""

    def __init__(self, dependency_chain: List[Type]) -> None:
        """
        Initialize with the dependency chain that caused the circular reference.

        Args:
        ----
            dependency_chain: List of types in the circular dependency chain

        """
        self.dependency_chain = dependency_chain
        chain_str = " -> ".join(t.__name__ for t in dependency_chain)
        super().__init__(f"Circular dependency detected: {chain_str}")


class LifecycleScope(enum.Enum):
    """Lifecycle scopes for container registrations."""

    SINGLETON = "singleton"  # One instance for the entire application
    TRANSIENT = "transient"  # New instance created each time
    SCOPED = "scoped"  # One instance per scope (e.g., request)


# Alias for backward compatibility
Scope = LifecycleScope


class SaplingsContainer:
    """
    Dependency injection container for Saplings.

    This container manages the lifecycle and dependencies of all services
    in the Saplings framework. It supports singleton, transient, and scoped
    registrations, and provides automatic dependency resolution.

    Features:
    - Circular dependency detection
    - Lazy initialization of singletons
    - Automatic constructor injection
    - Factory function support
    - Instance tracking for memory management
    """

    def __init__(self) -> None:
        """Initialize the container."""
        # Registrations map service types to their implementations
        self._registrations: Dict[Type, Any] = {}

        # Singleton instances are stored here
        self._singletons: Dict[Type, Any] = {}

        # Factory functions for creating instances
        self._factories: Dict[Type, Callable[..., Any]] = {}

        # Track which services are being resolved to detect circular dependencies
        self._resolution_stack: List[Type] = []

        # Thread lock for thread safety
        self._lock = threading.RLock()

        # Track all created instances for debugging
        self._instances: Set[weakref.ref] = set()

        logger.debug("Container initialized")

    def register(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type] = None,
        *,
        instance: Optional[T] = None,
        factory: Optional[Callable[..., T]] = None,
        scope: LifecycleScope = LifecycleScope.SINGLETON,
        **kwargs: Any,
    ) -> None:
        """
        Register a service with the container.

        Args:
        ----
            service_type: The interface or base type to register
            implementation_type: The concrete type to instantiate
            instance: A pre-created instance to use (for singletons)
            factory: A factory function to create instances
            scope: The lifecycle scope for the registration
            **kwargs: Additional arguments to pass to the constructor or factory

        """
        with self._lock:
            if instance is not None:
                # Register a specific instance as a singleton
                self._singletons[service_type] = instance
                logger.debug(f"Registered instance of {service_type.__name__}")
            elif factory is not None:
                # Register a factory function
                self._factories[service_type] = factory
                self._registrations[service_type] = (None, scope, kwargs)
                logger.debug(
                    f"Registered factory for {service_type.__name__} with scope {scope.value}"
                )
            else:
                # Register a type
                concrete_type = implementation_type or service_type
                self._registrations[service_type] = (concrete_type, scope, kwargs)
                logger.debug(
                    f"Registered {concrete_type.__name__} for {service_type.__name__} with scope {scope.value}"
                )

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service from the container.

        This method creates or retrieves an instance of the requested service type,
        automatically resolving any dependencies.

        Args:
        ----
            service_type: The type to resolve

        Returns:
        -------
            An instance of the requested type

        Raises:
        ------
            CircularDependencyError: If a circular dependency is detected
            KeyError: If the service type is not registered
            Exception: If the service cannot be created

        """
        with self._lock:
            # Check for circular dependencies
            if service_type in self._resolution_stack:
                # We've detected a circular dependency
                # Add the current type to complete the circle
                chain = self._resolution_stack + [service_type]
                raise CircularDependencyError(chain)

            # Check if we already have a singleton instance
            if service_type in self._singletons:
                return cast(T, self._singletons[service_type])

            # If not registered, try to register the type as its own implementation
            if service_type not in self._registrations and service_type not in self._factories:
                logger.debug(f"Auto-registering {service_type.__name__} as its own implementation")
                self.register(service_type)

            # Get the registration
            if service_type in self._factories:
                factory = self._factories[service_type]
                _, scope, kwargs = self._registrations[service_type]

                # Add to resolution stack to detect circular dependencies
                self._resolution_stack.append(service_type)

                try:
                    # Resolve any dependencies in kwargs
                    resolved_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, type):
                            resolved_kwargs[key] = self.resolve(value)
                        else:
                            resolved_kwargs[key] = value

                    # Create the instance
                    instance = factory(**resolved_kwargs)

                    # Store as singleton if requested
                    if scope == LifecycleScope.SINGLETON:
                        self._singletons[service_type] = instance

                    # Track the instance
                    if instance is not None:
                        self._track_instance(instance)

                    return cast(T, instance)
                finally:
                    # Remove from resolution stack
                    self._resolution_stack.pop()

            elif service_type in self._registrations:
                concrete_type, scope, kwargs = self._registrations[service_type]

                # Add to resolution stack to detect circular dependencies
                self._resolution_stack.append(service_type)

                try:
                    # Create the instance
                    instance = self._create_instance(concrete_type, **kwargs)

                    # Store as singleton if requested
                    if scope == LifecycleScope.SINGLETON:
                        self._singletons[service_type] = instance

                    # Track the instance
                    if instance is not None:
                        self._track_instance(instance)

                    return cast(T, instance)
                finally:
                    # Remove from resolution stack
                    self._resolution_stack.pop()

            # If we get here, we couldn't resolve the service
            msg = f"Service {service_type.__name__} is not registered"
            raise KeyError(msg)

    def _create_instance(self, concrete_type: Type[T], **kwargs: Any) -> T:
        """
        Create an instance of a type, resolving constructor dependencies.

        Args:
        ----
            concrete_type: The type to instantiate
            **kwargs: Additional arguments to pass to the constructor

        Returns:
        -------
            An instance of the requested type

        """
        import inspect

        # Get constructor parameters
        try:
            sig = inspect.signature(concrete_type.__init__)
            params = sig.parameters

            # Skip self parameter
            params = {k: v for k, v in params.items() if k != "self"}

            # Prepare constructor arguments
            constructor_args = {}

            # Add explicitly provided arguments
            constructor_args.update(kwargs)

            # Resolve remaining parameters from the container
            for name, param in params.items():
                if name not in constructor_args and param.annotation != inspect.Parameter.empty:
                    try:
                        constructor_args[name] = self.resolve(param.annotation)
                    except Exception:
                        # Skip parameters that can't be resolved
                        pass

            # Create the instance
            return concrete_type(**constructor_args)

        except Exception as e:
            # If we can't create the instance with dependencies, try without parameters
            try:
                return concrete_type()
            except Exception:
                msg = f"Failed to create instance of {concrete_type.__name__}: {e!s}"
                raise Exception(msg) from e

    def _track_instance(self, instance: Any) -> None:
        """
        Track an instance for memory management.

        Args:
        ----
            instance: The instance to track

        """
        # Create a weak reference to avoid memory leaks
        ref = weakref.ref(instance, self._cleanup_ref)
        self._instances.add(ref)

    def _cleanup_ref(self, ref: weakref.ref) -> None:
        """
        Clean up a weak reference when the object is garbage collected.

        Args:
        ----
            ref: The weak reference to clean up

        """
        with self._lock:
            if ref in self._instances:
                self._instances.remove(ref)

    def clear(self) -> None:
        """Clear all registrations and instances."""
        with self._lock:
            self._registrations.clear()
            self._singletons.clear()
            self._factories.clear()
            self._resolution_stack.clear()
            self._instances.clear()
            logger.debug("Container cleared")

    def get_instance_count(self) -> int:
        """
        Get the number of tracked instances.

        Returns
        -------
            The number of instances tracked by the container

        """
        with self._lock:
            # Filter out dead references
            live_instances = [ref for ref in self._instances if ref() is not None]
            return len(live_instances)
