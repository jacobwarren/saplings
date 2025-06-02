from __future__ import annotations

"""
Dependency injection module for Saplings.

This module provides a centralized dependency injection container
for managing service dependencies with constructor injection.
"""

# Import directly from the internal modules to avoid circular imports
from saplings.di._internal.container import Container, container
from saplings.di._internal.initialization import inject, register, reset_container

# Use the global container instance from the container module

# Re-export for backward compatibility
__all__ = ["Container", "container", "inject", "register", "reset_container"]
