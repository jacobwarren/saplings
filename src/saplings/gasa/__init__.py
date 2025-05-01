"""
Graph-Aligned Sparse Attention (GASA) module for Saplings.

This module provides the GASA functionality for Saplings, including:
- Graph traversal algorithm for token-level mask generation
- Mask serialization/deserialization
- Integration points for executor.py
- Block-diagonal packing fallback for unsupported models
- Visualization utilities for debugging masks

GASA injects a binary attention mask—derived from the retrieval dependency graph—into
each transformer layer, permitting full attention only between tokens whose source chunks
are ≤ h hops apart in the graph, while routing others through a lightweight global summary token.

This approach reduces computational cost (up to 40% fewer FLOPs) and improves grounding
by focusing the model's attention on relevant context.
"""

from saplings.gasa.block_packing import BlockDiagonalPacker
from saplings.gasa.config import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.gasa.mask_builder import MaskBuilder, MaskFormat, MaskType
from saplings.gasa.visualization import MaskVisualizer

__all__ = [
    "GASAConfig",
    "MaskBuilder",
    "MaskFormat",
    "MaskType",
    "MaskStrategy",
    "FallbackStrategy",
    "BlockDiagonalPacker",
    "MaskVisualizer",
]
