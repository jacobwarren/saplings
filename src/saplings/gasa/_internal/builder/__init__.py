from __future__ import annotations

"""
Builder module for Graph-Aligned Sparse Attention (GASA).

This module provides implementations of the MaskBuilderInterface for building
attention masks based on document dependency graphs.
"""


from saplings.gasa._internal.builder.standard_mask_builder import StandardMaskBuilder
from saplings.gasa._internal.builder.token_tracking_mask_builder import TokenTrackingMaskBuilder

# Re-export TokenTrackingMaskBuilder as StandardMaskBuilder for backward compatibility
# This makes TokenTrackingMaskBuilder the default implementation
StandardMaskBuilder = TokenTrackingMaskBuilder

__all__ = [
    "StandardMaskBuilder",
    "TokenTrackingMaskBuilder",
]
