from __future__ import annotations

"""
Agent configuration module.

This module provides the configuration class for Agent instances.
"""

import logging
import os
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class AgentConfig:
    """
    Configuration class for Agent instances.

    This class provides a structured way to configure Agent instances
    with various options and dependencies.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        memory_path: str = "./memory",
        output_dir: str = "./output",
        enable_gasa: bool = False,
        enable_monitoring: bool = True,
        enable_self_healing: bool = False,
        enable_tool_factory: bool = False,
        planner_budget_strategy: str = "fixed",
        planner_allow_budget_overflow: bool = False,
        planner_budget_overflow_margin: float = 0.1,
        executor_validation_type: str = "basic",
        tool_factory_sandbox_enabled: bool = True,
        allowed_imports: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        supported_modalities: Optional[List[str]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the agent configuration.

        Args:
        ----
            provider: The model provider (e.g., "openai", "anthropic", "vllm")
            model_name: The name of the model to use
            memory_path: Path to the memory store directory
            output_dir: Path to the output directory
            enable_gasa: Whether to enable Graph-Aligned Sparse Attention
            enable_monitoring: Whether to enable monitoring
            enable_self_healing: Whether to enable self-healing
            enable_tool_factory: Whether to enable the tool factory
            planner_budget_strategy: Strategy for allocating budget to tasks
            planner_allow_budget_overflow: Whether to allow budget overflow
            planner_budget_overflow_margin: Margin for budget overflow
            executor_validation_type: Type of validation to use for execution
            tool_factory_sandbox_enabled: Whether to enable sandboxing for tools
            allowed_imports: List of allowed imports for tools
            tools: List of tools to register with the agent
            supported_modalities: List of supported modalities
            model_parameters: Additional parameters for the model

        Raises:
        ------
            ValueError: If required parameters are missing or invalid
            OSError: If paths cannot be created

        """
        # Validate required parameters with helpful error messages
        self._validate_required_parameters(provider, model_name)
        self._validate_provider(provider)
        self._validate_paths(memory_path, output_dir)
        # Basic configuration
        self.provider = provider
        self.model_name = model_name
        self.memory_path = memory_path
        self.output_dir = output_dir
        self.model_parameters = model_parameters or {}

        # Feature flags
        self.enable_gasa = enable_gasa
        self.enable_monitoring = enable_monitoring
        self.enable_self_healing = enable_self_healing
        self.enable_tool_factory = enable_tool_factory

        # Planner configuration
        self.planner_budget_strategy = planner_budget_strategy
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
        supported_modalities_list = ["text", "image", "audio", "video"]
        for modality in self.supported_modalities:
            if modality not in supported_modalities_list:
                raise ValueError(
                    f"Unsupported modality '{modality}'. Supported modalities are: {', '.join(supported_modalities_list)}. "
                    f"Please specify valid modalities. "
                    f"Example: AgentConfig(provider='openai', model_name='gpt-4o', supported_modalities=['text', 'image'])"
                )

        # Ensure text is always supported
        if "text" not in self.supported_modalities:
            self.supported_modalities.append("text")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create memory directory
        os.makedirs(memory_path, exist_ok=True)

    def _validate_required_parameters(self, provider: str, model_name: str) -> None:
        """Validate that required parameters are provided with helpful error messages."""
        if not provider:
            raise ValueError(
                "The 'provider' parameter is required. Please specify a model provider such as "
                "'openai', 'anthropic', 'vllm', or 'huggingface'. "
                "Example: AgentConfig(provider='openai', model_name='gpt-4o')"
            )

        if not model_name:
            raise ValueError(
                "The 'model_name' parameter is required. Please specify the name of the model to use. "
                "Examples: 'gpt-4o' for OpenAI, 'claude-3-opus' for Anthropic, or 'Qwen/Qwen3-7B-Instruct' for vLLM. "
                "Example: AgentConfig(provider='openai', model_name='gpt-4o')"
            )

    def _validate_provider(self, provider: str) -> None:
        """Validate that the provider is supported with helpful error messages."""
        supported_providers = ["openai", "anthropic", "vllm", "huggingface", "test", "mock"]
        if provider.lower() not in supported_providers:
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported providers are: {', '.join(supported_providers)}. "
                f"Please choose one of the supported providers. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o')"
            )

    def _validate_paths(self, memory_path: str, output_dir: str) -> None:
        """Validate that paths are valid and can be created."""
        from pathlib import Path

        # Validate memory path
        try:
            memory_path_obj = Path(memory_path)
            if memory_path_obj.exists() and not memory_path_obj.is_dir():
                raise ValueError(
                    f"Memory path '{memory_path}' exists but is not a directory. "
                    f"Please specify a valid directory path for memory storage. "
                    f"Example: AgentConfig(provider='openai', model_name='gpt-4o', memory_path='./agent_memory')"
                )
        except (OSError, ValueError) as e:
            if "not a directory" in str(e):
                raise e
            raise ValueError(
                f"Invalid memory path '{memory_path}': {e}. "
                f"Please specify a valid directory path that can be created. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o', memory_path='./agent_memory')"
            ) from e

        # Validate output directory
        try:
            output_dir_obj = Path(output_dir)
            if output_dir_obj.exists() and not output_dir_obj.is_dir():
                raise ValueError(
                    f"Output directory '{output_dir}' exists but is not a directory. "
                    f"Please specify a valid directory path for output storage. "
                    f"Example: AgentConfig(provider='openai', model_name='gpt-4o', output_dir='./agent_output')"
                )
        except (OSError, ValueError) as e:
            if "not a directory" in str(e):
                raise e
            raise ValueError(
                f"Invalid output directory '{output_dir}': {e}. "
                f"Please specify a valid directory path that can be created. "
                f"Example: AgentConfig(provider='openai', model_name='gpt-4o', output_dir='./agent_output')"
            ) from e
