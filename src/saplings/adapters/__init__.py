from __future__ import annotations

"""Model adapters for Saplings.

This package provides adapter implementations for various LLM providers.

Note: This module re-exports the public API from saplings.api.models.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.models.

__all__ = [
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "VLLMAdapter",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        # Import from the public API
        if name == "AnthropicAdapter":
            from saplings.api.models import AnthropicAdapter

            return AnthropicAdapter
        elif name == "HuggingFaceAdapter":
            from saplings.api.models import HuggingFaceAdapter

            return HuggingFaceAdapter
        elif name == "OpenAIAdapter":
            from saplings.api.models import OpenAIAdapter

            return OpenAIAdapter
        elif name == "VLLMAdapter":
            from saplings.api.models import VLLMAdapter

            return VLLMAdapter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
