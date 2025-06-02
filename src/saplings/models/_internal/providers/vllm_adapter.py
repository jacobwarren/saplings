from __future__ import annotations

"""
vLLM adapter for Saplings.

This module provides an implementation of the LLM interface for vLLM models.
"""

# Re-export the adapter from the adapters component
from saplings.adapters._internal.providers.vllm_adapter import VLLMAdapter

__all__ = ["VLLMAdapter"]
