from __future__ import annotations

"""
Packing module for Graph-Aligned Sparse Attention (GASA).

This module provides implementations for token/chunk reordering strategies
that enable GASA functionality on models that don't natively support
sparse attention masks.
"""


from saplings.gasa._internal.packing.block_diagonal_packer import BlockDiagonalPacker

__all__ = [
    "BlockDiagonalPacker",
]
