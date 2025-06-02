from __future__ import annotations

"""
Patch generator for Saplings.

This module provides the public API for generating patches for self-healing.
"""

from saplings.self_heal._internal.patch_generator import PatchGenerator as _PatchGenerator

# Re-export the PatchGenerator class
PatchGenerator = _PatchGenerator

__all__ = ["PatchGenerator"]
