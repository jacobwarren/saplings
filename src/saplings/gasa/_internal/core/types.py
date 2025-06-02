from __future__ import annotations

"""
Type definitions for Graph-Aligned Sparse Attention (GASA).

This module provides enums and type definitions used throughout the GASA module.
"""


from enum import Enum


class MaskFormat(str, Enum):
    """Format of attention masks."""

    DENSE = "dense"  # Dense matrix (numpy array)
    SPARSE = "sparse"  # Sparse matrix (scipy.sparse)
    BLOCK_SPARSE = "block_sparse"  # Block-sparse format (list of blocks)


class MaskType(str, Enum):
    """Type of attention masks."""

    ATTENTION = "attention"  # Regular attention mask (0 = masked, 1 = attend)
    GLOBAL_ATTENTION = "global_attention"  # Global attention mask (1 = global attention)
