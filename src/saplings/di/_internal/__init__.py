from __future__ import annotations

"""
Internal implementation of the dependency injection system.

This package contains the internal implementation details of the
dependency injection system. These modules should not be imported
directly by users of the library.
"""

# Import from individual modules
from saplings.di._internal.container import Container, container
from saplings.di._internal.initialization import inject, register, reset_container

# Import from registration subdirectory
from saplings.di._internal.registration import (
    ConfiguredProvider,
    FactoryProvider,
    InitializableProvider,
    LazyProvider,
    Provider,
    SingletonProvider,
)

__all__ = [
    # Core components
    "Container",
    "container",
    "inject",
    "register",
    "reset_container",
    # Registration components
    "Provider",
    "FactoryProvider",
    "SingletonProvider",
    "ConfiguredProvider",
    "LazyProvider",
    "InitializableProvider",
]
