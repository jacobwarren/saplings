from __future__ import annotations

"""
Internal tokenizers module for Saplings API.

This module provides internal constants and utilities for tokenizers.
"""


# Check if transformers is available for shadow model tokenizer without importing
def _check_transformers_available() -> bool:
    """Check if transformers is available without importing it."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("transformers")
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


SHADOW_MODEL_AVAILABLE = _check_transformers_available()

__all__ = ["SHADOW_MODEL_AVAILABLE"]
