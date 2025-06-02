from __future__ import annotations

"""
Core utilities for Saplings.

This package provides stateless helper functions and utilities with
no external dependencies outside of the core modules.
"""

from saplings.core._internal.utils.platform import (
    get_platform_info,
    is_apple_silicon,
    is_linux,
    is_macos,
    is_triton_available,
    is_windows,
)
from saplings.core._internal.utils.tokenizer import (
    count_tokens,
    get_tokens_remaining,
    split_text_by_tokens,
    truncate_text_tokens,
)

__all__ = [
    "count_tokens",
    "get_tokens_remaining",
    "split_text_by_tokens",
    "truncate_text_tokens",
    "get_platform_info",
    "is_apple_silicon",
    "is_linux",
    "is_macos",
    "is_triton_available",
    "is_windows",
]
