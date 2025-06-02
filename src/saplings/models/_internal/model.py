from __future__ import annotations

"""
Model interface module.

This module provides the core model interfaces and base classes for the Saplings framework.
"""

# Direct imports from the models component
from saplings.models._internal.interfaces import (
    LLM,
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
)
from saplings.models._internal.llm_builder import LLMBuilder

__all__ = [
    "LLM",
    "LLMBuilder",
    "LLMResponse",
    "ModelCapability",
    "ModelMetadata",
    "ModelRole",
]
