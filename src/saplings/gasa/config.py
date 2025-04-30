"""
Configuration module for Graph-Aligned Sparse Attention (GASA).

This module defines the configuration classes for the GASA module.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MaskStrategy(str, Enum):
    """Strategy for applying attention masks."""
    
    BINARY = "binary"  # Binary mask (0/1)
    SOFT = "soft"  # Soft mask (continuous values between 0 and 1)
    LEARNED = "learned"  # Learned mask (requires fine-tuning)


class FallbackStrategy(str, Enum):
    """Fallback strategy for models that don't support sparse attention."""
    
    BLOCK_DIAGONAL = "block_diagonal"  # Reorder tokens into block-diagonal structure
    DENSE = "dense"  # Fall back to dense attention
    WINDOWED = "windowed"  # Use sliding window attention


class GASAConfig(BaseModel):
    """Configuration for Graph-Aligned Sparse Attention (GASA)."""
    
    enabled: bool = Field(
        True, description="Whether to enable GASA"
    )
    max_hops: int = Field(
        2, description="Maximum number of hops for attention (h parameter)"
    )
    mask_strategy: MaskStrategy = Field(
        MaskStrategy.BINARY, description="Strategy for applying attention masks"
    )
    fallback_strategy: FallbackStrategy = Field(
        FallbackStrategy.BLOCK_DIAGONAL, 
        description="Fallback strategy for models that don't support sparse attention"
    )
    global_tokens: List[str] = Field(
        ["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
        description="Tokens that should attend to all other tokens"
    )
    summary_token: str = Field(
        "[SUM]", description="Token used for global summary"
    )
    add_summary_token: bool = Field(
        True, description="Whether to add a summary token if not present"
    )
    block_size: int = Field(
        512, description="Block size for block-diagonal packing"
    )
    overlap: int = Field(
        64, description="Overlap between blocks for block-diagonal packing"
    )
    soft_mask_temperature: float = Field(
        0.1, description="Temperature for soft masks (lower = closer to binary)"
    )
    cache_masks: bool = Field(
        True, description="Whether to cache generated masks"
    )
    cache_dir: Optional[str] = Field(
        None, description="Directory to cache masks"
    )
    visualize: bool = Field(
        False, description="Whether to generate visualizations"
    )
    visualization_dir: Optional[str] = Field(
        None, description="Directory to save visualizations"
    )
    
    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> "GASAConfig":
        """
        Create a configuration from command-line arguments.
        
        Args:
            args: Command-line arguments
            
        Returns:
            GASAConfig: Configuration
        """
        config = cls()
        
        if "gasa" in args:
            config.enabled = args["gasa"]
        
        if "gasa_hop" in args:
            config.max_hops = args["gasa_hop"]
        
        if "gasa_strategy" in args:
            config.mask_strategy = MaskStrategy(args["gasa_strategy"])
        
        if "gasa_fallback" in args:
            config.fallback_strategy = FallbackStrategy(args["gasa_fallback"])
        
        if "gasa_block_size" in args:
            config.block_size = args["gasa_block_size"]
        
        if "gasa_overlap" in args:
            config.overlap = args["gasa_overlap"]
        
        if "gasa_cache" in args:
            config.cache_masks = args["gasa_cache"]
        
        if "gasa_cache_dir" in args:
            config.cache_dir = args["gasa_cache_dir"]
        
        if "gasa_visualize" in args:
            config.visualize = args["gasa_visualize"]
        
        if "gasa_visualization_dir" in args:
            config.visualization_dir = args["gasa_visualization_dir"]
        
        return config
