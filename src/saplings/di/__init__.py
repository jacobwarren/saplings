from __future__ import annotations

"""
Dependency injection container for Saplings.

This module re-exports the public API from saplings.api.container.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides a centralized dependency injection container
for managing service dependencies with constructor injection.
"""

# Import directly from internal modules to avoid circular imports
from saplings.di._internal import (
    Container,
    container,
    inject,
    register,
    reset_container,
)

# Import provider types from internal implementation
# These are not yet in the public API
from saplings.di._internal.registration import (
    ConfiguredProvider,
    FactoryProvider,
    LazyProvider,
    Provider,
    SingletonProvider,
)


# Use the new container state management from api.di
def configure_container(config, context_id: str = "default"):
    """
    Configure the container with the given configuration.

    Args:
    ----
        config: The configuration to use for the container.
        context_id: Context identifier for configuration isolation.

    """
    # Delegate to the main API implementation
    from saplings.api.di import configure_container as _configure_container

    return _configure_container(config, context_id)


def reset_container_config():
    """
    Reset the container configuration to its default state.
    """
    # Delegate to the main API implementation
    from saplings.api.di import reset_container_config as _reset_container_config

    return _reset_container_config()


def reset_container_context(context_id: str = "default"):
    """
    Reset container context safely.

    Args:
    ----
        context_id: Context identifier to reset

    """
    # Delegate to the main API implementation
    from saplings.api.di import reset_container_context as _reset_container_context

    return _reset_container_context(context_id)


__all__ = [
    "Container",
    "container",
    "inject",
    "register",
    "reset_container",
    "configure_container",
    "reset_container_config",
    "reset_container_context",
    # Provider types
    "Provider",
    "FactoryProvider",
    "SingletonProvider",
    "ConfiguredProvider",
    "LazyProvider",
]
