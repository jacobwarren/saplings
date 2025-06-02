from __future__ import annotations

"""
vLLM fallback adapter for Saplings.

This module provides a fallback implementation of the LLM interface for vLLM models
when the vLLM package is not available.
"""

# Re-export the adapter from the adapters component
from saplings.adapters._internal.providers.vllm_fallback_adapter import VLLMFallbackAdapter

__all__ = ["VLLMFallbackAdapter"]
