from __future__ import annotations

"""
Model registry for Saplings.

This module provides a registry for LLM instances to ensure only one instance
of a model with the same configuration is created, which helps reduce memory
usage and improve performance.
"""

from saplings.core._internal.model_registry import (
    ModelKey,
    ModelRegistry,
    create_model_key,
    get_model_registry,
)

__all__ = [
    "ModelRegistry",
    "ModelKey",
    "create_model_key",
    "get_model_registry",
]
