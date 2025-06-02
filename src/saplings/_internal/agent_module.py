from __future__ import annotations

"""
Agent configuration module for Saplings.

This module provides the AgentConfig class that centralizes all configuration
options for the Agent. It serves as the internal implementation for the public API.
"""

import logging
import os
from typing import Any, ClassVar, Dict

# Configure logging
logger = logging.getLogger(__name__)


class AgentConfig:
    """
    Configuration for the Agent and AgentFacade classes.

    This class centralizes all configuration options,
    making it easy to customize behavior while providing sensible defaults.

    For common use cases, use the preset factory methods:
    - `minimal()`: Minimal configuration with only essential features
    - `standard()`: Standard configuration with balanced features
    - `full_featured()`: Full-featured configuration with all features enabled
    - `for_openai()`: Configuration optimized for OpenAI models
    - `for_anthropic()`: Configuration optimized for Anthropic models
    - `for_vllm()`: Configuration optimized for vLLM models
    """

    # Configuration presets as class variables
    PRESET_MINIMAL: ClassVar[Dict[str, Any]] = {
        "enable_gasa": False,
        "enable_monitoring": False,
        "enable_self_healing": False,
        "enable_tool_factory": False,
        "planner_budget_strategy": "fixed",
        "executor_validation_type": "execution",
    }

    PRESET_STANDARD: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "enable_monitoring": True,
        "enable_self_healing": False,
        "enable_tool_factory": True,
        "planner_budget_strategy": "token_count",
        "executor_validation_type": "execution",
    }

    PRESET_FULL_FEATURED: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "enable_monitoring": True,
        "enable_self_healing": True,
        "enable_tool_factory": True,
        "gasa_max_hops": 3,
        "retrieval_max_documents": 20,
        "planner_budget_strategy": "dynamic",
        "planner_total_budget": 2.0,
        "planner_allow_budget_overflow": True,
        "executor_validation_type": "judge",
    }

    PRESET_OPENAI: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "gasa_shadow_model": True,
        "gasa_shadow_model_name": "Qwen/Qwen3-0.6B",
        "gasa_fallback": "prompt_composer",
        "gasa_prompt_composer": True,
        "max_tokens": 4096,
    }

    PRESET_ANTHROPIC: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "gasa_shadow_model": True,
        "gasa_shadow_model_name": "Qwen/Qwen3-0.6B",
        "gasa_fallback": "prompt_composer",
        "gasa_prompt_composer": True,
        "max_tokens": 4096,
    }

    PRESET_VLLM: ClassVar[Dict[str, Any]] = {
        "enable_gasa": True,
        "gasa_shadow_model": False,
        "gasa_fallback": "block_diagonal",
        "model_parameters": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "trust_remote_code": True,
        },
    }

    def __init__(
        self,
        provider: str,
        model_name: str,
        memory_path: str = "./agent_memory",
        output_dir: str = "./agent_output",
        enable_gasa: bool = True,
        enable_monitoring: bool = True,
        enable_self_healing: bool = True,
        self_healing_max_retries: int = 3,
        enable_tool_factory: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        gasa_max_hops: int = 2,
        gasa_strategy: str = "binary",
        gasa_fallback: str = "block_diagonal",
        gasa_shadow_model: bool = False,
        gasa_shadow_model_name: str = "Qwen/Qwen3-0.6B",
        gasa_prompt_composer: bool = False,
        retrieval_entropy_threshold: float = 0.1,
        retrieval_max_documents: int = 10,
        planner_budget_strategy: str = "proportional",
        planner_total_budget: float = 1.0,
        planner_allow_budget_overflow: bool = False,
        planner_budget_overflow_margin: float = 0.1,
        executor_validation_type: str = "judge",
        tool_factory_sandbox_enabled: bool = True,
        allowed_imports: list[str] | None = None,
        tools: list[Any] | None = None,
        supported_modalities: list[str] | None = None,
        **model_parameters,
    ) -> None:
        """
        Initialize the agent configuration.

        Args:
        ----
            provider: Model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: Model name
            memory_path: Path to store agent memory
            output_dir: Directory to save outputs
            enable_gasa: Whether to enable Graph-Aligned Sparse Attention
            enable_monitoring: Whether to enable monitoring and tracing
            enable_self_healing: Whether to enable self-healing capabilities
            self_healing_max_retries: Maximum number of retry attempts for self-healing operations
            enable_tool_factory: Whether to enable dynamic tool creation
            max_tokens: Maximum number of tokens for model responses
            temperature: Temperature for model generation
            gasa_max_hops: Maximum number of hops for GASA mask
            gasa_strategy: Strategy for GASA mask (binary, soft, learned)
            gasa_fallback: Fallback strategy for models that don't support sparse attention
            gasa_shadow_model: Whether to enable shadow model for tokenization
            gasa_shadow_model_name: Name of the shadow model to use
            gasa_prompt_composer: Whether to enable graph-aware prompt composer
            retrieval_entropy_threshold: Entropy threshold for retrieval termination
            retrieval_max_documents: Maximum number of documents to retrieve
            planner_budget_strategy: Strategy for budget allocation
            planner_total_budget: Total budget for the planner in USD
            planner_allow_budget_overflow: Whether to allow exceeding the budget
            planner_budget_overflow_margin: Margin by which the budget can be exceeded (as a fraction)
            executor_validation_type: Strategy for output validation
            tool_factory_sandbox_enabled: Whether to enable sandbox for tool execution
            allowed_imports: List of allowed imports for dynamic tools
            tools: List of tools to register with the agent
            supported_modalities: List of supported modalities (text, image, audio, video)
            **model_parameters: Additional model parameters

        """
        # Store provider and model_name as primary fields
        self.provider = provider
        self.model_name = model_name
        self._model_parameters = model_parameters
        self.memory_path = memory_path
        self.output_dir = output_dir
        self.enable_gasa = enable_gasa
        self.enable_monitoring = enable_monitoring
        self.enable_self_healing = enable_self_healing
        self.self_healing_max_retries = self_healing_max_retries
        self.enable_tool_factory = enable_tool_factory
        self.max_tokens = max_tokens
        self.temperature = temperature

        # GASA configuration
        self.gasa_max_hops = gasa_max_hops
        self.gasa_strategy = gasa_strategy
        self.gasa_fallback = gasa_fallback
        self.gasa_shadow_model = gasa_shadow_model
        self.gasa_shadow_model_name = gasa_shadow_model_name
        self.gasa_prompt_composer = gasa_prompt_composer

        # Retrieval configuration
        self.retrieval_entropy_threshold = retrieval_entropy_threshold
        self.retrieval_max_documents = retrieval_max_documents

        # Planner configuration
        self.planner_budget_strategy = planner_budget_strategy
        self.planner_total_budget = planner_total_budget
        self.planner_allow_budget_overflow = planner_allow_budget_overflow
        self.planner_budget_overflow_margin = planner_budget_overflow_margin

        # Executor configuration
        self.executor_validation_type = executor_validation_type

        # Tool factory configuration
        self.tool_factory_sandbox_enabled = tool_factory_sandbox_enabled
        self.allowed_imports = allowed_imports or [
            "os",
            "datetime",
            "json",
            "math",
            "numpy",
            "pandas",
        ]
        self.tools = tools or []

        # Modality support
        self.supported_modalities = supported_modalities or ["text"]

        # Validate supported modalities
        for modality in self.supported_modalities:
            if modality not in ["text", "image", "audio", "video"]:
                msg = f"Unsupported modality: {modality}"
                raise ValueError(msg)

        # Ensure text is always supported
        if "text" not in self.supported_modalities:
            self.supported_modalities.append("text")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create memory directory
        os.makedirs(memory_path, exist_ok=True)

    @property
    def model_parameters(self):
        """
        Get model parameters.

        Returns
        -------
            Dict[str, Any]: Model parameters

        """
        return self._model_parameters

    @classmethod
    def minimal(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a minimal configuration with only essential features enabled.

        This configuration is optimized for simplicity and performance, with most
        advanced features disabled.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Minimal configuration

        """
        # Start with minimal preset
        config_params = cls.PRESET_MINIMAL.copy()

        # Add required parameters
        config_params["provider"] = provider
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def standard(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a standard configuration with balanced features.

        This configuration is suitable for most use cases, with a good balance
        between features and performance.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Standard configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add required parameters
        config_params["provider"] = provider
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def full_featured(cls, provider: str, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a full-featured configuration with all features enabled.

        This configuration enables all advanced features for maximum capability,
        but may be more resource-intensive.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Full-featured configuration

        """
        # Start with full-featured preset
        config_params = cls.PRESET_FULL_FEATURED.copy()

        # Add required parameters
        config_params["provider"] = provider
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def for_openai(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for OpenAI models.

        This configuration includes settings that work well with OpenAI models,
        such as appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: OpenAI-optimized configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add OpenAI-specific settings
        config_params.update(cls.PRESET_OPENAI)

        # Add required parameters
        config_params["provider"] = "openai"
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def for_anthropic(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for Anthropic models.

        This configuration includes settings that work well with Anthropic models,
        such as appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: Anthropic model name (e.g., "claude-3-opus", "claude-3-sonnet")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: Anthropic-optimized configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add Anthropic-specific settings
        config_params.update(cls.PRESET_ANTHROPIC)

        # Add required parameters
        config_params["provider"] = "anthropic"
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)

    @classmethod
    def for_vllm(cls, model_name: str, **kwargs) -> AgentConfig:
        """
        Create a configuration optimized for vLLM models.

        This configuration includes settings that work well with vLLM models,
        such as appropriate GASA settings for self-hosted models.

        Args:
        ----
            model_name: Model name (e.g., "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentConfig: vLLM-optimized configuration

        """
        # Start with standard preset
        config_params = cls.PRESET_STANDARD.copy()

        # Add vLLM-specific settings
        config_params.update(cls.PRESET_VLLM)

        # Add required parameters
        config_params["provider"] = "vllm"
        config_params["model_name"] = model_name

        # Override with any provided kwargs
        config_params.update(kwargs)

        return cls(**config_params)
