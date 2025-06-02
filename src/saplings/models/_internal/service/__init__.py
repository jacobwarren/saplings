from __future__ import annotations

"""
Service module for model components.

This module provides service classes for models in the Saplings framework.
"""

# Use lazy imports to avoid circular dependencies
from importlib import import_module


def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    service_mapping = {
        "ModelInitializationService": "saplings.services._internal.model_initialization_service",
        "ModelCachingService": "saplings.services._internal.model_caching_service",
        "ModelInitializationServiceBuilder": "saplings.services._internal.builders.model_initialization_service_builder",
    }

    if name in service_mapping:
        module = import_module(service_mapping[name])
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelInitializationService",
    "ModelCachingService",
    "ModelInitializationServiceBuilder",
]
