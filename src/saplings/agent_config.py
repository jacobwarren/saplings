from __future__ import annotations

"""
Agent configuration module for Saplings.

This module provides the AgentConfig class which centralizes all configuration
options for the Agent and AgentFacade classes, making it easy to customize
behavior while providing sensible defaults.
"""


import os
from typing import Any


class AgentConfig:
    """
    Configuration for the Agent and AgentFacade classes.

    This class centralizes all configuration options,
    making it easy to customize behavior while providing sensible defaults.
    """

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
        planner_budget_strategy: str = "token_count",
        planner_total_budget: float = 1.0,
        planner_allow_budget_overflow: bool = False,
        planner_budget_overflow_margin: float = 0.1,
        executor_verification_strategy: str = "judge",
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
            executor_verification_strategy: Strategy for output verification
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
        self.executor_verification_strategy = executor_verification_strategy

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
