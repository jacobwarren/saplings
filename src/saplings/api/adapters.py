from __future__ import annotations

"""
Adapters API module for Saplings.

This module provides the public API for model adapters.
"""

# Import lightweight adapters directly
from saplings.adapters._internal.providers import (
    AnthropicAdapter as _AnthropicAdapter,
)
from saplings.adapters._internal.providers import (
    OpenAIAdapter as _OpenAIAdapter,
)


# Lazy import heavy adapters
def _get_heavy_adapter(name: str):
    """Get heavy adapter using lazy import."""
    from saplings.adapters._internal.providers import __getattr__ as providers_getattr

    return providers_getattr(name)


from saplings.api.stability import beta, stable


@stable
class OpenAIAdapter(_OpenAIAdapter):
    """
    Adapter for OpenAI models.

    This adapter provides an interface to OpenAI models, handling authentication,
    request formatting, and response parsing.
    """


@stable
class AnthropicAdapter(_AnthropicAdapter):
    """
    Adapter for Anthropic models.

    This adapter provides an interface to Anthropic models, handling authentication,
    request formatting, and response parsing.
    """


# Heavy adapters are created lazily
def __getattr__(name: str):
    """Lazy import heavy adapters using centralized system."""
    if name == "HuggingFaceAdapter":
        _HuggingFaceAdapter = _get_heavy_adapter("HuggingFaceAdapter")

        @beta
        class HuggingFaceAdapter(_HuggingFaceAdapter):
            """
            Adapter for Hugging Face models.

            This adapter provides an interface to Hugging Face models, handling model loading,
            tokenization, and generation.
            """

        return HuggingFaceAdapter

    elif name == "VLLMAdapter":
        _VLLMAdapter = _get_heavy_adapter("VLLMAdapter")

        @beta
        class VLLMAdapter(_VLLMAdapter):
            """
            Adapter for vLLM models.

            This adapter provides an interface to vLLM models, handling model loading,
            tokenization, and generation with optimized inference.
            """

        return VLLMAdapter

    elif name == "TransformersAdapter":
        _TransformersAdapter = _get_heavy_adapter("TransformersAdapter")

        @beta
        class TransformersAdapter(_TransformersAdapter):
            """
            Adapter for Transformers models.

            This adapter provides an interface to Transformers models, handling model loading,
            tokenization, and generation.
            """

        return TransformersAdapter

    elif name == "VLLMFallbackAdapter":
        _VLLMFallbackAdapter = _get_heavy_adapter("VLLMFallbackAdapter")

        @beta
        class VLLMFallbackAdapter(_VLLMFallbackAdapter):
            """
            Adapter for vLLM models with fallback.

            This adapter provides an interface to vLLM models with fallback to other
            adapters if vLLM is not available or fails.
            """

        return VLLMFallbackAdapter

    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "TransformersAdapter",
    "VLLMAdapter",
    "VLLMFallbackAdapter",
]
