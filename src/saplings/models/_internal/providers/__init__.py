from __future__ import annotations

"""
Providers module for model components.

This module provides model provider implementations for the Saplings framework.
"""

# Use lazy imports to avoid circular dependencies
from importlib import import_module


def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    adapter_mapping = {
        "AnthropicAdapter": "saplings.models._internal.providers.anthropic_adapter",
        "HuggingFaceAdapter": "saplings.models._internal.providers.huggingface_adapter",
        "OpenAIAdapter": "saplings.models._internal.providers.openai_adapter",
        "VLLMAdapter": "saplings.models._internal.providers.vllm_adapter",
        "TransformersAdapter": "saplings.models._internal.providers.transformers_adapter",
        "VLLMFallbackAdapter": "saplings.models._internal.providers.vllm_fallback_adapter",
        "LazyInitializable": "saplings.models._internal.providers.lazy_initializable",
        "ModelAdapterFactory": "saplings.models._internal.providers.model_adapter_factory",
    }

    if name in adapter_mapping:
        module = import_module(adapter_mapping[name])
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "VLLMAdapter",
    "TransformersAdapter",
    "VLLMFallbackAdapter",
    "LazyInitializable",
    "ModelAdapterFactory",
]
