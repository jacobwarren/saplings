from __future__ import annotations

"""
OpenAI adapter for Saplings.

This module provides an implementation of the LLM interface for OpenAI's models.
"""

# Re-export the adapter from the adapters component
from saplings.adapters._internal.providers.openai_adapter import OpenAIAdapter

__all__ = ["OpenAIAdapter"]
