from __future__ import annotations

"""
Provider type wrappers for the DI container.

This module defines typed provider wrappers that can be used
to register more complex dependencies in the container.
"""

from typing import Callable, List, Optional, Type, TypeVar

# Import the Provider interface
from saplings.di._internal.registration.interfaces import Provider

T = TypeVar("T")


class SingletonProvider(Provider[T]):
    """Provider that returns the same instance every time."""

    def __init__(self, instance: T):
        self._instance = instance

    def get(self) -> T:
        return self._instance


class FactoryProvider(Provider[T]):
    """Provider that creates a new instance each time."""

    def __init__(self, factory: Callable[..., T], **kwargs):
        self._factory = factory
        self._kwargs = kwargs

    def get(self) -> T:
        return self._factory(**self._kwargs)


class ConfiguredProvider(Provider[T]):
    """Provider that configures an instance before returning it."""

    def __init__(self, provider: Provider[T], configurator: Callable[[T], None]):
        self._provider = provider
        self._configurator = configurator

    def get(self) -> T:
        instance = self._provider.get()
        self._configurator(instance)
        return instance


class LazyProvider(Provider[T]):
    """
    Provider that lazily initializes the instance.

    This provider defers the creation of the instance until it is actually needed.
    It can be used to break initialization cycles by providing a way to create
    instances of services without resolving their dependencies immediately.

    Example:
    -------
    ```python
    # Register a service with a lazy dependency
    container.register(
        ServiceA,
        factory=lambda: ServiceA(service_b_factory=lambda: container.resolve(ServiceB))
    )
    ```

    """

    def __init__(self, provider_factory: Callable[[], Provider[T]]):
        """
        Initialize the lazy provider.

        Args:
        ----
            provider_factory: A factory function that returns a provider

        """
        self._provider_factory = provider_factory
        self._provider: Optional[Provider[T]] = None

    def get(self) -> T:
        """
        Get the instance.

        This method creates the provider and gets the instance from it
        the first time it is called. Subsequent calls return the same
        instance from the cached provider.

        Returns
        -------
            The instance

        """
        if self._provider is None:
            self._provider = self._provider_factory()
        return self._provider.get()


class InitializableProvider(Provider[T]):
    """
    Provider that supports initialization tracking.

    This provider wraps another provider and adds initialization tracking.
    It can be used to ensure that services are properly initialized before
    they are used.

    Example:
    -------
    ```python
    # Register a service with initialization tracking
    container.register(
        ServiceA,
        provider=InitializableProvider(
            SingletonProvider(ServiceAImpl()),
            dependencies=[ServiceB, ServiceC]
        )
    )
    ```

    """

    def __init__(
        self,
        provider: Provider[T],
        dependencies: Optional[List[Type]] = None,
        initialize_method: str = "initialize",
    ):
        """
        Initialize the initializable provider.

        Args:
        ----
            provider: The provider to wrap
            dependencies: The types that this service depends on for initialization
            initialize_method: The name of the method to call for initialization

        """
        self._provider = provider
        self._dependencies = dependencies or []
        self._initialize_method = initialize_method
        self._initialized = False

    def _get_container(self):
        """
        Get the container instance.

        This method imports the container at runtime to avoid circular imports.

        Returns
        -------
            The container instance

        """

        # Import here to avoid circular imports at module level
        # Use a function to delay the import until runtime
        def get_container():
            # Use importlib to avoid circular imports
            import importlib

            container_module = importlib.import_module("saplings.di._internal.container")
            return container_module.container

        return get_container()

    def get(self) -> T:
        """
        Get the instance.

        This method gets the instance from the wrapped provider and
        initializes it if it hasn't been initialized yet.

        Returns
        -------
            The instance

        """
        # Get the instance from the wrapped provider
        instance = self._provider.get()

        # Initialize the instance if it hasn't been initialized yet
        if not self._initialized:
            # Initialize dependencies first
            for dependency_type in self._dependencies:
                # Get the container instance without importing it directly
                # This avoids circular imports at module level
                container = self._get_container()

                # Check if the dependency is registered
                if (
                    dependency_type in container._registrations
                    or dependency_type in container._factories
                ):
                    # Initialize the dependency
                    container.initialize_service(dependency_type)

            # Call the initialize method if it exists
            if hasattr(instance, self._initialize_method) and callable(
                getattr(instance, self._initialize_method)
            ):
                getattr(instance, self._initialize_method)()

            self._initialized = True

        return instance


__all__ = [
    "Provider",
    "FactoryProvider",
    "SingletonProvider",
    "ConfiguredProvider",
    "LazyProvider",
    "InitializableProvider",
]
