from __future__ import annotations

"""
Patches module for self-healing components.

This module provides patch functionality for the Saplings framework.
"""

from saplings.self_heal._internal.patches.patch import (
    Patch,
    PatchStatus,
)
from saplings.self_heal._internal.patches.patch_generator import (
    PatchGenerator,
    PatchResult,
)

__all__ = [
    "Patch",
    "PatchStatus",
    "PatchGenerator",
    "PatchResult",
]
