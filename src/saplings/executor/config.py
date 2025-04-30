"""
Configuration module for the Executor.

This module defines the configuration classes for the Executor module.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class VerificationStrategy(str, Enum):
    """Strategy for verifying generated outputs."""
    
    NONE = "none"  # No verification
    BASIC = "basic"  # Basic verification with simple checks
    JUDGE = "judge"  # Verification using JudgeAgent
    VALIDATOR = "validator"  # Verification using ValidatorRegistry
    FULL = "full"  # Full verification using both JudgeAgent and ValidatorRegistry


class RefinementStrategy(str, Enum):
    """Strategy for refining rejected outputs."""
    
    NONE = "none"  # No refinement
    RETRY = "retry"  # Simple retry with the same prompt
    FEEDBACK = "feedback"  # Retry with feedback from verification
    ITERATIVE = "iterative"  # Iterative refinement with multiple feedback cycles


class ExecutorConfig(BaseModel):
    """Configuration for the Executor module."""
    
    # Speculative execution settings
    enable_speculative_execution: bool = Field(
        True, description="Whether to enable speculative execution"
    )
    draft_temperature: float = Field(
        0.2, description="Temperature for draft generation"
    )
    final_temperature: float = Field(
        0.7, description="Temperature for final generation"
    )
    max_draft_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens for draft generation"
    )
    max_final_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens for final generation"
    )
    
    # Streaming settings
    enable_streaming: bool = Field(
        True, description="Whether to enable streaming output"
    )
    stream_chunk_size: int = Field(
        10, description="Number of tokens to generate per streaming chunk"
    )
    
    # GASA settings
    enable_gasa: bool = Field(
        True, description="Whether to enable GASA"
    )
    gasa_config: Optional[Dict[str, Any]] = Field(
        None, description="GASA configuration"
    )
    
    # Verification settings
    verification_strategy: VerificationStrategy = Field(
        VerificationStrategy.BASIC, description="Strategy for verifying generated outputs"
    )
    verification_threshold: float = Field(
        0.7, description="Threshold for verification (0.0 to 1.0)"
    )
    
    # Refinement settings
    refinement_strategy: RefinementStrategy = Field(
        RefinementStrategy.FEEDBACK, description="Strategy for refining rejected outputs"
    )
    max_refinement_attempts: int = Field(
        3, description="Maximum number of refinement attempts"
    )
    
    # Performance settings
    cache_results: bool = Field(
        True, description="Whether to cache results"
    )
    cache_dir: Optional[str] = Field(
        None, description="Directory to cache results"
    )
    
    # Logging settings
    log_level: str = Field(
        "INFO", description="Logging level"
    )
    
    @classmethod
    def default(cls) -> "ExecutorConfig":
        """
        Create a default configuration.
        
        Returns:
            ExecutorConfig: Default configuration
        """
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExecutorConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExecutorConfig: Configuration
        """
        return cls(**config_dict)
