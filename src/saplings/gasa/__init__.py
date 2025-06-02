from __future__ import annotations

"""
Graph-Aligned Sparse Attention (GASA) module for Saplings.

This module provides the core GASA functionality for the Saplings framework.
For application code, it is recommended to import from the top-level saplings package.

GASA injects a binary attention mask—derived from the retrieval dependency graph—into
each transformer layer, permitting full attention only between tokens whose source chunks
are ≤ h hops apart in the graph, while routing others through a lightweight global summary token.

This approach reduces computational cost (up to 40% fewer FLOPs) and improves grounding
by focusing the model's attention on relevant context.

For third-party LLM APIs like OpenAI and Anthropic that don't expose low-level attention
mechanisms, GASA provides alternative approaches:
1. Shadow model tokenization - Uses a small local model for tokenization and mask generation
2. Graph-aware prompt composition - Structures prompts based on graph relationships
3. Block-diagonal packing - Reorders chunks to create a block-diagonal structure
"""

# Import directly from internal modules to avoid circular imports
from saplings.gasa._internal.config import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.gasa._internal.core.types import MaskFormat, MaskType
from saplings.gasa._internal.gasa_config_builder import GASAConfigBuilder
from saplings.gasa._internal.packing.block_diagonal_packer import BlockDiagonalPacker
from saplings.gasa._internal.packing.block_pack import block_pack
from saplings.gasa._internal.builder.standard_mask_builder import StandardMaskBuilder
from saplings.gasa._internal.builder.token_tracking_mask_builder import TokenTrackingMaskBuilder
from saplings.gasa._internal.core.graph_distance import GraphDistanceCalculator
from saplings.gasa._internal.core.token_mapper import TokenMapper
from saplings.gasa._internal.prompt.prompt_composer import GASAPromptComposer
from saplings.gasa._internal.service.gasa_service import GASAService
from saplings.gasa._internal.service.gasa_service_builder import GASAServiceBuilder
from saplings.gasa._internal.service.null_gasa_service import NullGASAService
from saplings.gasa._internal.visualization.mask_visualizer import MaskVisualizer

# For backward compatibility
GASABlockDiagonalPacker = BlockDiagonalPacker

# Import shadow model tokenizer if available
try:
    from saplings.tokenizers.shadow_model_tokenizer import ShadowModelTokenizer

    SHADOW_MODEL_AVAILABLE = True
except ImportError:
    SHADOW_MODEL_AVAILABLE = False

__all__ = [
    "BlockDiagonalPacker",
    "FallbackStrategy",
    "GASABlockDiagonalPacker",
    "GASAConfig",
    "GASAConfigBuilder",
    "GASAPromptComposer",
    "GASAService",
    "GASAServiceBuilder",
    "GraphDistanceCalculator",
    "MaskFormat",
    "MaskStrategy",
    "MaskType",
    "MaskVisualizer",
    "NullGASAService",
    "StandardMaskBuilder",
    "TokenMapper",
    "TokenTrackingMaskBuilder",
    "block_pack",
]

# Add shadow model tokenizer to exports if available
if SHADOW_MODEL_AVAILABLE:
    __all__.append("ShadowModelTokenizer")
