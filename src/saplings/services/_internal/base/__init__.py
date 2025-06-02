from __future__ import annotations

"""
Base service module for services components.

This module provides base classes for service implementations in the Saplings framework.
"""

from saplings.services._internal.base.lazy_service import LazyService
from saplings.services._internal.base.lazy_service_builder import LazyServiceBuilder
from saplings.services._internal.base.service_dependency_graph import ServiceDependencyGraph

__all__ = [
    "LazyService",
    "LazyServiceBuilder",
    "ServiceDependencyGraph",
]
