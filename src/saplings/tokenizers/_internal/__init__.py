from __future__ import annotations

"""
Internal implementation of the Tokenizers module.

This module provides the implementation of tokenizer components for the Saplings framework.
"""

from saplings.tokenizers._internal.factory import TokenizerFactory
from saplings.tokenizers._internal.simple_tokenizer import SimpleTokenizer


# Check shadow model availability without importing
def _check_shadow_model_available() -> bool:
    """Check if shadow model tokenizer is available without importing it."""
    import importlib.util

    try:
        torch_spec = importlib.util.find_spec("torch")
        transformers_spec = importlib.util.find_spec("transformers")
        return torch_spec is not None and transformers_spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


SHADOW_MODEL_AVAILABLE = _check_shadow_model_available()


# Lazy import for shadow model tokenizer
def __getattr__(name: str):
    """Lazy import shadow model tokenizer to avoid loading heavy dependencies."""
    if name == "ShadowModelTokenizer" and SHADOW_MODEL_AVAILABLE:
        from saplings.tokenizers._internal.shadow import ShadowModelTokenizer

        return ShadowModelTokenizer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base tokenizer
    "SimpleTokenizer",
    "TokenizerFactory",
]

# Add shadow model tokenizer to exports if available
if SHADOW_MODEL_AVAILABLE:
    __all__.append("ShadowModelTokenizer")
