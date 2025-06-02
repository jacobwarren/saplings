from __future__ import annotations

"""
Utility modules for Saplings.

This module provides utility functions for Saplings.
"""

from saplings.core._internal.utils.platform import (
    get_platform_info,
    is_linux,
    is_macos,
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
    "get_platform_info",
    "get_tokens_remaining",
    "is_linux",
    "is_macos",
    "is_windows",
    "split_text_by_tokens",
    "truncate_text_tokens",
]
