from __future__ import annotations

"""
Core components and interfaces for Graph-Aligned Sparse Attention (GASA).

This module provides the fundamental types, interfaces, and base classes
for the GASA module.
"""


from saplings.gasa._internal.core.chunk_info import ChunkInfo
from saplings.gasa._internal.core.interfaces import MaskBuilderInterface
from saplings.gasa._internal.core.types import MaskFormat, MaskType

__all__ = [
    "ChunkInfo",
    "MaskBuilderInterface",
    "MaskFormat",
    "MaskType",
]
