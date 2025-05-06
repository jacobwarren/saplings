from __future__ import annotations

"""Model adapters for Saplings.

This package provides adapter implementations for various LLM providers.
"""

# Standard library imports
import contextlib  # noqa: E402

# Local imports
with contextlib.suppress(ImportError):
    from saplings.adapters.anthropic_adapter import AnthropicAdapter

with contextlib.suppress(ImportError):
    from saplings.adapters.huggingface_adapter import HuggingFaceAdapter

with contextlib.suppress(ImportError):
    from saplings.adapters.openai_adapter import OpenAIAdapter

with contextlib.suppress(ImportError):
    from saplings.adapters.vllm_adapter import VLLMAdapter

__all__ = [
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "VLLMAdapter",
]
