from __future__ import annotations

"""
Tokenizers module for Saplings.

This module re-exports the public API from saplings.api.tokenizers.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides tokenizer implementations for various models,
including a simple tokenizer and a shadow model tokenizer for GASA.
It also provides a factory for creating tokenizers on-demand.
"""


# Define SHADOW_MODEL_AVAILABLE using lazy checking to avoid eager imports
def _check_transformers_available() -> bool:
    """Check if transformers is available without importing it."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("transformers")
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


SHADOW_MODEL_AVAILABLE = _check_transformers_available()

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.tokenizers.

__all__ = ["SimpleTokenizer", "TokenizerFactory", "SHADOW_MODEL_AVAILABLE"]

# Add shadow model tokenizer to exports if available
if SHADOW_MODEL_AVAILABLE:
    __all__.append("ShadowModelTokenizer")


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.tokenizers import (
            SimpleTokenizer,
            TokenizerFactory,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "SimpleTokenizer": SimpleTokenizer,
            "TokenizerFactory": TokenizerFactory,
        }

        # Add shadow model tokenizer if available
        if SHADOW_MODEL_AVAILABLE and name == "ShadowModelTokenizer":
            from saplings.api.tokenizers import ShadowModelTokenizer

            globals_dict["ShadowModelTokenizer"] = ShadowModelTokenizer

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
