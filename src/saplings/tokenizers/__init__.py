from __future__ import annotations

"""
Tokenizers module for Saplings.

This module provides tokenizer implementations for various models,
including a simple tokenizer and a shadow model tokenizer for GASA.
"""


from .simple_tokenizer import SimpleTokenizer

# Import shadow model tokenizer if available
try:
    from .shadow_model_tokenizer import ShadowModelTokenizer

    SHADOW_MODEL_AVAILABLE = True
except ImportError:
    SHADOW_MODEL_AVAILABLE = False

__all__ = ["SimpleTokenizer"]

# Add shadow model tokenizer to exports if available
if SHADOW_MODEL_AVAILABLE:
    __all__.append("ShadowModelTokenizer")
