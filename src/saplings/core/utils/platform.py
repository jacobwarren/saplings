from __future__ import annotations

"""
Platform-specific utilities.

This module provides platform-specific utility functions.
"""

from saplings.core._internal.utils.platform import (
    get_platform_info,
    is_apple_silicon,
    is_linux,
    is_macos,
    is_triton_available,
    is_windows,
)

__all__ = [
    "get_platform_info",
    "is_apple_silicon",
    "is_linux",
    "is_macos",
    "is_triton_available",
    "is_windows",
]
