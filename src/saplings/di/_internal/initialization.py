from __future__ import annotations

"""
Initialization module for dependency injection in Saplings.

This module provides functions for initializing and resetting the DI container.
"""

import logging
from typing import Callable, TypeVar

from saplings.di._internal.container import container
from saplings.di._internal.inject import inject as _inject

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic type hints
T = TypeVar("T")


def register(cls=None, /, **kwargs):
    """
    Register a class or instance with the container.

    This function can be used as a decorator or called directly.

    When used as a decorator:
        @register
        class MyClass: ...

        @register(IService)
        class ServiceImpl: ...

    When called directly:
        register(IService, instance=my_instance)
        register(IService, factory=lambda: ServiceImpl())
        register(IService, scope=LifecycleScope.SCOPED)

    Args:
    ----
        cls: The class to register or the interface to register for
        **kwargs: Registration options
            - instance: A specific instance to register
            - factory: A factory function to create instances
            - singleton: Whether to register as singleton (default: True)
            - scope: The lifecycle scope for the registration
            - Any other kwargs are passed to the factory function

    Returns:
    -------
        The decorator function or the concrete class

    """
    # Import here to avoid circular imports

    def decorator(concrete_cls):
        # Extract registration options
        instance = kwargs.get("instance")
        factory_func = kwargs.get("factory")
        singleton = kwargs.get("singleton", True)
        scope = kwargs.get("scope")
        provider = kwargs.get("provider")
        dependencies = kwargs.get("dependencies")

        # Remove our special kwargs so they don't get passed to the factory
        registration_kwargs = kwargs.copy()
        if "instance" in registration_kwargs:
            del registration_kwargs["instance"]
        if "factory" in registration_kwargs:
            del registration_kwargs["factory"]
        if "singleton" in registration_kwargs:
            del registration_kwargs["singleton"]
        if "scope" in registration_kwargs:
            del registration_kwargs["scope"]
        if "provider" in registration_kwargs:
            del registration_kwargs["provider"]
        if "dependencies" in registration_kwargs:
            del registration_kwargs["dependencies"]

        # Determine the service type
        service_type = cls if cls is not None and cls != concrete_cls else concrete_cls

        # Register based on the provided options
        if provider is not None:
            container.register(
                service_type,
                provider=provider,
                dependencies=dependencies,
                singleton=singleton,
                scope=scope,
            )
        elif instance is not None:
            container.register(service_type, instance=instance, dependencies=dependencies)
        elif factory_func is not None:
            container.register(
                service_type,
                factory=factory_func,
                singleton=singleton,
                scope=scope,
                dependencies=dependencies,
                **registration_kwargs,
            )
        else:
            container.register(
                service_type,
                concrete_type=concrete_cls,
                singleton=singleton,
                scope=scope,
                dependencies=dependencies,
            )

        return concrete_cls

    # Handle different calling patterns
    if cls is None:
        # Called as @register() or register()
        return decorator
    if isinstance(cls, type):
        # Called as @register
        if cls == register:
            return decorator
        # Called as @register(IService)
        if not any([kwargs.get("instance"), kwargs.get("factory")]):
            return decorator
    # Called as register(IService, instance=x) or similar
    return decorator(cls)


def inject(func: Callable) -> Callable:
    """
    Decorator to auto-resolve constructor args.

    This decorator automatically resolves dependencies for a function
    based on its type annotations. It supports both regular type annotations
    and Inject[T] annotations for explicit dependency marking.

    Args:
    ----
        func: The function to inject dependencies into

    Returns:
    -------
        Wrapped function with auto-resolved dependencies

    """
    # Use the inject function from the inject module
    return _inject(func)


def reset_container():
    """
    Reset the container to its initial state.

    This is primarily used in tests to ensure isolation between test runs.
    """
    container.clear()
    logger.debug("Container reset")
