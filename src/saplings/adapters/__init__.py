"""
Model adapters for Saplings.

This package provides implementations of the LLM interface for various providers.
"""

# Import adapters if available
try:
    from saplings.adapters.vllm_adapter import VLLMAdapter
except ImportError:
    pass

try:
    from saplings.adapters.openai_adapter import OpenAIAdapter
except ImportError:
    pass

try:
    from saplings.adapters.anthropic_adapter import AnthropicAdapter
except ImportError:
    pass

try:
    from saplings.adapters.huggingface_adapter import HuggingFaceAdapter
except ImportError:
    pass

__all__ = [
    "VLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HuggingFaceAdapter",
]
