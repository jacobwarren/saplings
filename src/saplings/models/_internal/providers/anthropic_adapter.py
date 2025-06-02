from __future__ import annotations

"""
Anthropic adapter for Saplings.

This module provides an implementation of the LLM interface for Anthropic's Claude models.
"""

# Re-export the adapter from the adapters component
from saplings.adapters._internal.providers.anthropic_adapter import AnthropicAdapter

__all__ = ["AnthropicAdapter"]
