from __future__ import annotations

"""
Registration module for dependency injection.

This module provides registration functionality for the DI container.
"""

# Import provider interface
from saplings.di._internal.registration.interfaces import Provider

# Import provider implementations
from saplings.di._internal.registration.providers import (
    ConfiguredProvider,
    FactoryProvider,
    InitializableProvider,
    LazyProvider,
    SingletonProvider,
)

__all__ = [
    # Provider interface
    "Provider",
    # Provider implementations
    "FactoryProvider",
    "SingletonProvider",
    "ConfiguredProvider",
    "LazyProvider",
    "InitializableProvider",
]
