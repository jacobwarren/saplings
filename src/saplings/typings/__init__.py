"""
Typings module for Saplings.

This module provides type definitions and stubs for optional dependencies.
"""

from __future__ import annotations

from saplings.typings._internal import (
    # Document types
    DocumentDict,
    # Common types
    JsonDict,
    JsonValue,
    MetadataDict,
    # Model types
    ModelConfig,
    ModelResponse,
)

__all__ = [
    # Common types
    "JsonDict",
    "JsonValue",
    # Document types
    "DocumentDict",
    "MetadataDict",
    # Model types
    "ModelConfig",
    "ModelResponse",
]
