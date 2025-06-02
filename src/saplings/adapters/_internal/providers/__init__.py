from __future__ import annotations

"""
Providers module for adapter components.

This module provides adapter implementations for different providers in the Saplings framework.
"""

# Import lightweight adapters directly
# Use centralized lazy import system
from saplings._internal.lazy_imports import lazy_import
from saplings._internal.optional_deps import OPTIONAL_DEPENDENCIES
from saplings.adapters._internal.providers.anthropic_adapter import AnthropicAdapter
from saplings.adapters._internal.providers.openai_adapter import OpenAIAdapter

# Create lazy importers for heavy adapters
_heavy_adapters = {
    "HuggingFaceAdapter": lazy_import(
        "saplings.adapters._internal.providers.huggingface_adapter",
        "HuggingFace adapter requires transformers. Install with: pip install saplings[gasa]",
        OPTIONAL_DEPENDENCIES.get("transformers"),
    ),
    "TransformersAdapter": lazy_import(
        "saplings.adapters._internal.providers.transformers_adapter",
        "Transformers adapter requires transformers. Install with: pip install saplings[gasa]",
        OPTIONAL_DEPENDENCIES.get("transformers"),
    ),
    "VLLMAdapter": lazy_import(
        "saplings.adapters._internal.providers.vllm_adapter",
        "vLLM adapter requires vLLM. Install with: pip install saplings[vllm]",
        OPTIONAL_DEPENDENCIES.get("vllm"),
    ),
    "VLLMFallbackAdapter": lazy_import(
        "saplings.adapters._internal.providers.vllm_fallback_adapter",
        "vLLM fallback adapter requires vLLM. Install with: pip install saplings[vllm]",
        OPTIONAL_DEPENDENCIES.get("vllm"),
    ),
}


def __getattr__(name: str):
    """Lazy import heavy adapters using centralized system."""
    if name in _heavy_adapters:
        lazy_importer = _heavy_adapters[name]
        # Get the actual class from the module
        module = lazy_importer._get_module()
        return getattr(module, name)
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
