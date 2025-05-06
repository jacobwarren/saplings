from __future__ import annotations

"""
Builder module for Graph-Aligned Sparse Attention (GASA).

This module provides implementations of the MaskBuilderInterface for building
attention masks based on document dependency graphs.
"""


from saplings.gasa.builder.standard_mask_builder import StandardMaskBuilder

__all__ = [
    "StandardMaskBuilder",
]
