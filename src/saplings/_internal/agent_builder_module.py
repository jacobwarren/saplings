from __future__ import annotations

"""
Agent Builder module for Saplings.

This module provides a builder class for creating Agent instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import Any, Dict, List

# Configure logging
logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Builder for Agent instances.

    This class provides a fluent interface for building Agent instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    The builder supports two initialization modes:
    1. Container-based initialization (default): Uses the dependency injection container
       for full flexibility and advanced features.
    2. Direct initialization: Creates services directly without container configuration,
       which simplifies the initialization flow for basic use cases.
    """

    def __init__(self) -> None:
        """Initialize the agent builder with default values."""
        self._config_params: Dict[str, Any] = {
            "provider": None,
            "model_name": None,
            "memory_path": "./agent_memory",
            "output_dir": "./agent_output",
            "enable_gasa": True,
            "enable_monitoring": True,
            "enable_self_healing": True,
            "self_healing_max_retries": 3,
            "enable_tool_factory": True,
            "max_tokens": 2048,
            "temperature": 0.7,
            "gasa_max_hops": 2,
            "gasa_strategy": "binary",
            "gasa_fallback": "block_diagonal",
            "gasa_shadow_model": False,
            "gasa_shadow_model_name": "Qwen/Qwen3-0.6B",
            "gasa_prompt_composer": False,
            "retrieval_entropy_threshold": 0.1,
            "retrieval_max_documents": 10,
            "planner_budget_strategy": "proportional",
            "planner_total_budget": 1.0,
            "planner_allow_budget_overflow": False,
            "planner_budget_overflow_margin": 0.1,
            "executor_validation_type": "execution",
            "tool_factory_sandbox_enabled": True,
            "allowed_imports": ["os", "json", "re", "math", "datetime", "time", "random"],
            "tools": [],
            "supported_modalities": ["text"],
            "model_parameters": {},
        }
        self._required_params = ["provider", "model_name"]

    def with_provider(self, provider: str) -> AgentBuilder:
        """
        Set the model provider.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["provider"] = provider
        return self

    def with_model_name(self, model_name: str) -> AgentBuilder:
        """
        Set the model name.

        Args:
        ----
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["model_name"] = model_name
        return self

    def with_memory_path(self, memory_path: str) -> AgentBuilder:
        """
        Set the memory path.

        Args:
        ----
            memory_path: Path to store agent memory

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["memory_path"] = memory_path
        return self

    def with_output_dir(self, output_dir: str) -> AgentBuilder:
        """
        Set the output directory.

        Args:
        ----
            output_dir: Directory to store agent output

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["output_dir"] = output_dir
        return self

    def with_gasa_enabled(self, enabled: bool) -> AgentBuilder:
        """
        Enable or disable GASA.

        Args:
        ----
            enabled: Whether to enable GASA

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["enable_gasa"] = enabled
        return self

    def with_monitoring_enabled(self, enabled: bool) -> AgentBuilder:
        """
        Enable or disable monitoring.

        Args:
        ----
            enabled: Whether to enable monitoring

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["enable_monitoring"] = enabled
        return self

    def with_tools(self, tools: List[Any]) -> AgentBuilder:
        """
        Set the tools to register with the agent.

        Args:
        ----
            tools: List of tools to register

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["tools"] = tools
        return self

    def with_model_parameters(self, parameters: Dict[str, Any]) -> AgentBuilder:
        """
        Set additional model parameters.

        Args:
        ----
            parameters: Model parameters

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params["model_parameters"] = parameters
        return self

    def with_config(self, config: Dict[str, Any]) -> AgentBuilder:
        """
        Update configuration with a dictionary.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config_params.update(config)
        return self

    @classmethod
    def minimal(cls, provider: str, model_name: str, **kwargs) -> AgentBuilder:
        """
        Create a builder with minimal configuration.

        This creates a builder with a minimal configuration preset that has only
        essential features enabled, optimized for simplicity and performance.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentBuilder: Builder with minimal configuration

        """
        # Import here to avoid circular imports
        from saplings._internal.agent_module import AgentConfig

        # Create a new builder
        builder = cls()

        # Create a minimal config
        config = AgentConfig.minimal(provider, model_name, **kwargs)

        # Update the builder's config params with all values from the config
        for key, value in vars(config).items():
            if not key.startswith("_"):  # Skip private attributes
                builder._config_params[key] = value

        # Add model parameters
        builder._config_params["model_parameters"] = config.model_parameters

        return builder

    @classmethod
    def standard(cls, provider: str, model_name: str, **kwargs) -> AgentBuilder:
        """
        Create a builder with standard configuration.

        This creates a builder with a standard configuration preset that has a good
        balance between features and performance, suitable for most use cases.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentBuilder: Builder with standard configuration

        """
        # Import here to avoid circular imports
        from saplings._internal.agent_module import AgentConfig

        # Create a new builder
        builder = cls()

        # Create a standard config
        config = AgentConfig.standard(provider, model_name, **kwargs)

        # Update the builder's config params with all values from the config
        for key, value in vars(config).items():
            if not key.startswith("_"):  # Skip private attributes
                builder._config_params[key] = value

        # Add model parameters
        builder._config_params["model_parameters"] = config.model_parameters

        return builder

    @classmethod
    def full_featured(cls, provider: str, model_name: str, **kwargs) -> AgentBuilder:
        """
        Create a builder with full-featured configuration.

        This creates a builder with a full-featured configuration preset that has
        all advanced features enabled, but may be more resource-intensive.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentBuilder: Builder with full-featured configuration

        """
        # Import here to avoid circular imports
        from saplings._internal.agent_module import AgentConfig

        # Create a new builder
        builder = cls()

        # Create a full-featured config
        config = AgentConfig.full_featured(provider, model_name, **kwargs)

        # Update the builder's config params with all values from the config
        for key, value in vars(config).items():
            if not key.startswith("_"):  # Skip private attributes
                builder._config_params[key] = value

        # Add model parameters
        builder._config_params["model_parameters"] = config.model_parameters

        return builder

    def build(self) -> "Agent":
        """
        Build the Agent instance.

        Returns
        -------
            Configured Agent instance

        Raises
        ------
            ValueError: If required configuration is missing

        """
        # Import here to avoid circular imports
        from saplings._internal.agent_class import Agent
        from saplings._internal.agent_module import AgentConfig

        # Validate required fields
        if not self._config_params.get("provider"):
            raise ValueError("Provider is required")
        if not self._config_params.get("model_name"):
            raise ValueError("Model name is required")

        # Create config from builder state
        config = AgentConfig(**self._config_params)
        return Agent(config)
