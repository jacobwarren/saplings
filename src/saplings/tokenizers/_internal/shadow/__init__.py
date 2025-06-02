from __future__ import annotations

"""
Shadow model module for tokenizer components.

This module provides shadow model tokenizer functionality for the Saplings framework.
"""


# Lazy import to avoid loading heavy dependencies during basic import
def __getattr__(name: str):
    """Lazy import shadow model tokenizer to avoid loading heavy dependencies."""
    if name == "ShadowModelTokenizer":
        from saplings.tokenizers._internal.shadow.shadow_model_tokenizer import ShadowModelTokenizer

        return ShadowModelTokenizer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ShadowModelTokenizer",
]
