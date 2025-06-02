from __future__ import annotations

"""
Agent configuration module for Saplings.

This module provides the AgentConfig class for configuring Agent instances.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Configuration for Agent instances.

    This class provides a structured way to configure Agent instances
    with various options and dependencies.
    """

    # Required parameters
    provider: str
    model_name: str

    # Optional parameters with defaults
    memory_path: str = "./agent_memory"
    output_dir: str = "./agent_output"
    enable_gasa: bool = True
    enable_monitoring: bool = True
    enable_self_healing: bool = True
    enable_tool_factory: bool = True
    max_tokens: int = 2048
    temperature: float = 0.7
    gasa_max_hops: int = 2
    gasa_strategy: str = "binary"
    gasa_fallback: str = "block_diagonal"
    gasa_shadow_model: bool = False
    gasa_shadow_model_name: str = "Qwen/Qwen3-0.6B"
    gasa_prompt_composer: bool = False
    retrieval_entropy_threshold: float = 0.1
    retrieval_max_documents: int = 10
    planner_budget_strategy: str = "proportional"
    planner_total_budget: float = 1.0
    planner_allow_budget_overflow: bool = False
    planner_budget_overflow_margin: float = 0.1
    executor_validation_type: str = "execution"
    tool_factory_sandbox_enabled: bool = True
    allowed_imports: List[str] = field(
        default_factory=lambda: ["os", "json", "re", "math", "datetime", "time", "random"]
    )
    tools: List[Any] = field(default_factory=list)
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    self_healing_max_retries: int = 3

    @classmethod
    def minimal(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a minimal configuration.

        This creates a configuration with only essential features enabled,
        optimized for simplicity and performance.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Configuration with minimal features

        """
        config = cls(
            provider=provider,
            model_name=model_name,
            enable_gasa=False,
            enable_monitoring=False,
            enable_self_healing=False,
            enable_tool_factory=False,
            **kwargs,
        )
        return config

    @classmethod
    def standard(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a standard configuration.

        This creates a configuration with a good balance between features and performance,
        suitable for most use cases.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Configuration with standard features

        """
        config = cls(
            provider=provider,
            model_name=model_name,
            enable_gasa=True,
            enable_monitoring=True,
            enable_self_healing=False,
            enable_tool_factory=True,
            **kwargs,
        )
        return config

    @classmethod
    def full_featured(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a full-featured configuration.

        This creates a configuration with all advanced features enabled,
        but may be more resource-intensive.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Configuration with all features

        """
        config = cls(
            provider=provider,
            model_name=model_name,
            enable_gasa=True,
            enable_monitoring=True,
            enable_self_healing=True,
            enable_tool_factory=True,
            **kwargs,
        )
        return config

    @classmethod
    def for_openai(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for OpenAI models.

        This creates a configuration optimized for OpenAI models,
        including appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Configuration optimized for OpenAI

        """
        config = cls(
            provider="openai",
            model_name=model_name,
            enable_gasa=True,
            gasa_strategy="binary",
            gasa_fallback="block_diagonal",
            gasa_shadow_model=True,
            gasa_shadow_model_name="Qwen/Qwen3-0.6B",
            **kwargs,
        )
        return config

    @classmethod
    def for_anthropic(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for Anthropic models.

        This creates a configuration optimized for Anthropic models,
        including appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: Anthropic model name (e.g., "claude-3-opus", "claude-3-sonnet")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Configuration optimized for Anthropic

        """
        config = cls(
            provider="anthropic",
            model_name=model_name,
            enable_gasa=True,
            gasa_strategy="binary",
            gasa_fallback="block_diagonal",
            gasa_shadow_model=True,
            gasa_shadow_model_name="Qwen/Qwen3-0.6B",
            **kwargs,
        )
        return config

    @classmethod
    def for_vllm(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for vLLM models.

        This creates a configuration optimized for vLLM models,
        including appropriate GASA settings for self-hosted models.

        Args:
        ----
            model_name: Model name (e.g., "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Configuration optimized for vLLM

        """
        config = cls(
            provider="vllm",
            model_name=model_name,
            enable_gasa=True,
            gasa_strategy="binary",
            gasa_fallback="none",  # No fallback needed for vLLM
            gasa_shadow_model=False,  # No shadow model needed for vLLM
            **kwargs,
        )
        return config
