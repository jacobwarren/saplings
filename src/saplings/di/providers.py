from __future__ import annotations

"""
Provider type wrappers for the DI container.

This module defines typed provider wrappers that can be used
to register more complex dependencies in the container.
"""


from typing import Callable, Generic, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


class Provider(Generic[T]):
    """Base provider interface for dependencies."""

    def provide(self):
        """Provide an instance of the dependency."""
        raise NotImplementedError


class FactoryProvider(Provider[T]):
    """Provider that uses a factory function to create instances."""

    def __init__(self, factory: Callable[..., T], **kwargs) -> None:
        """
        Initialize with a factory function and its arguments.

        Args:
        ----
            factory: A callable that creates the dependency
            **kwargs: Arguments to pass to the factory

        """
        self._factory = factory
        self._kwargs = kwargs

    def provide(self):
        """Create an instance using the factory."""
        return self._factory(**self._kwargs)


class SingletonProvider(Provider[T]):
    """Provider that ensures only one instance is created."""

    def __init__(self, provider: Provider[T]) -> None:
        """
        Initialize with an underlying provider.

        Args:
        ----
            provider: The provider to use for instance creation

        """
        self._provider = provider
        self._instance: T | None = None

    def provide(self):
        """Get or create the singleton instance."""
        if self._instance is None:
            self._instance = self._provider.provide()
        return self._instance


class ConfiguredProvider(Provider[T]):
    """Provider that configures an instance after creation."""

    def __init__(self, provider: Provider[T], configurator: Callable[[T], None]) -> None:
        """
        Initialize with a provider and a configurator.

        Args:
        ----
            provider: The provider to use for instance creation
            configurator: A function that configures the instance

        """
        self._provider = provider
        self._configurator = configurator

    def provide(self):
        """Create and configure an instance."""
        instance = self._provider.provide()
        self._configurator(instance)
        return instance


class LazyProvider(Provider[T]):
    """Provider that defers creation until first access."""

    def __init__(self, provider: Provider[T]) -> None:
        """
        Initialize with an underlying provider.

        Args:
        ----
            provider: The provider to use for instance creation

        """
        self._provider = provider
        self._initialized = False
        self._instance: T | None = None

    def provide(self):
        """Get or create the instance on first access."""
        if not self._initialized:
            self._instance = self._provider.provide()
            self._initialized = True
        return cast("T", self._instance)
