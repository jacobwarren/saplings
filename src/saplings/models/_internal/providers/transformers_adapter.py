from __future__ import annotations

"""
Transformers adapter for Saplings.

This module provides an implementation of the LLM interface for Transformers models.
"""

# Re-export the adapter from the adapters component
from saplings.adapters._internal.providers.transformers_adapter import TransformersAdapter

__all__ = ["TransformersAdapter"]
