from __future__ import annotations

"""
Function registry for Saplings.

This module provides a registry for function definitions that can be used
with function calling capabilities of LLMs.
"""

from saplings.core._internal.function_registry import (
    FunctionRegistry,
    get_function_registry,
)

__all__ = [
    "FunctionRegistry",
    "get_function_registry",
]
