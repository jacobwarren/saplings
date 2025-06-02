from __future__ import annotations

"""
Configuration module for model components.

This module provides configuration classes for models in the Saplings framework.
"""

# Use lazy imports to avoid circular dependencies
from importlib import import_module


def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    config_mapping = {
        "ModelMetadata": "saplings.core._internal.model_interface",
    }

    if name in config_mapping:
        module = import_module(config_mapping[name])
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelMetadata",
]
