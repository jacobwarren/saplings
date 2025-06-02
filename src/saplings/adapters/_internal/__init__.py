from __future__ import annotations

"""
Internal module for model adapters.

This module provides the implementation of model adapters for various providers.
"""

# Import from base
from saplings.adapters._internal.base import LazyInitializable

# Import lightweight providers directly
from saplings.adapters._internal.providers import (
    AnthropicAdapter,
    OpenAIAdapter,
)


# Lazy import for heavy providers
def __getattr__(name: str):
    """Lazy import heavy adapters to avoid loading dependencies during basic import."""
    if name in ["HuggingFaceAdapter", "TransformersAdapter", "VLLMAdapter", "VLLMFallbackAdapter"]:
        from saplings.adapters._internal.providers import __getattr__ as providers_getattr

        return providers_getattr(name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Import from utils
from saplings.adapters._internal.utils import ModelAdapterFactory

__all__ = [
    # Base
    "LazyInitializable",
    # Providers
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "TransformersAdapter",
    "VLLMAdapter",
    "VLLMFallbackAdapter",
    # Utils
    "ModelAdapterFactory",
]
