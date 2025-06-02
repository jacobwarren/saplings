from __future__ import annotations

"""
HuggingFace adapter for Saplings.

This module provides an implementation of the LLM interface for HuggingFace's models.
"""

# Re-export the adapter from the adapters component
from saplings.adapters._internal.providers.huggingface_adapter import HuggingFaceAdapter

__all__ = ["HuggingFaceAdapter"]
