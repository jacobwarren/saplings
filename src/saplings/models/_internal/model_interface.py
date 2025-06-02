from __future__ import annotations

"""
Model interface module for Saplings.

This module provides base interfaces for model adapters to avoid circular imports.
It re-exports the interfaces from the models component to avoid duplication.
"""

# Import directly from the interfaces module
from saplings.models._internal.interfaces import (
    LLM,
    LazyInitializable,
    LLMResponse,
    ModelAdapterFactory,
    ModelCapability,
    ModelMetadata,
    ModelRole,
)

# Re-export the interfaces
__all__ = [
    "LLM",
    "LLMResponse",
    "ModelCapability",
    "ModelMetadata",
    "ModelRole",
    "LazyInitializable",
    "ModelAdapterFactory",
]
