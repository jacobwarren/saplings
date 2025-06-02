from __future__ import annotations

"""
Agent Builder module part 2 for Saplings.

This module provides additional methods for the AgentBuilder class.
"""

import logging

# Configure logging
logger = logging.getLogger(__name__)


def add_provider_specific_methods(AgentBuilder):
    """Add provider-specific methods to the AgentBuilder class."""

    @classmethod
    def for_openai(cls, model_name: str, **kwargs) -> AgentBuilder:
        """
        Create a builder optimized for OpenAI models.

        This creates a builder with a configuration preset optimized for OpenAI models,
        including appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentBuilder: Builder optimized for OpenAI

        """
        # Import here to avoid circular imports
        from saplings._internal.agent_module import AgentConfig

        # Create a new builder
        builder = cls()

        # Create an OpenAI-optimized config
        config = AgentConfig.for_openai(model_name, **kwargs)

        # Update the builder's config params with all values from the config
        for key, value in vars(config).items():
            if not key.startswith("_"):  # Skip private attributes
                builder._config_params[key] = value

        # Add model parameters
        builder._config_params["model_parameters"] = config.model_parameters

        return builder

    @classmethod
    def for_anthropic(cls, model_name: str, **kwargs) -> AgentBuilder:
        """
        Create a builder optimized for Anthropic models.

        This creates a builder with a configuration preset optimized for Anthropic models,
        including appropriate GASA settings for API-based models.

        Args:
        ----
            model_name: Anthropic model name (e.g., "claude-3-opus", "claude-3-sonnet")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentBuilder: Builder optimized for Anthropic

        """
        # Import here to avoid circular imports
        from saplings._internal.agent_module import AgentConfig

        # Create a new builder
        builder = cls()

        # Create an Anthropic-optimized config
        config = AgentConfig.for_anthropic(model_name, **kwargs)

        # Update the builder's config params with all values from the config
        for key, value in vars(config).items():
            if not key.startswith("_"):  # Skip private attributes
                builder._config_params[key] = value

        # Add model parameters
        builder._config_params["model_parameters"] = config.model_parameters

        return builder

    @classmethod
    def for_vllm(cls, model_name: str, **kwargs) -> AgentBuilder:
        """
        Create a builder optimized for vLLM models.

        This creates a builder with a configuration preset optimized for vLLM models,
        including appropriate GASA settings for self-hosted models.

        Args:
        ----
            model_name: Model name (e.g., "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
            **kwargs: Additional configuration parameters to override defaults

        Returns:
        -------
            AgentBuilder: Builder optimized for vLLM

        """
        # Import here to avoid circular imports
        from saplings._internal.agent_module import AgentConfig

        # Create a new builder
        builder = cls()

        # Create a vLLM-optimized config
        config = AgentConfig.for_vllm(model_name, **kwargs)

        # Update the builder's config params with all values from the config
        for key, value in vars(config).items():
            if not key.startswith("_"):  # Skip private attributes
                builder._config_params[key] = value

        # Add model parameters
        builder._config_params["model_parameters"] = config.model_parameters

        return builder

    def _validate_config(self) -> None:
        """
        Validate that all required configuration parameters are provided.

        Raises
        ------
            InitializationError: If a required parameter is missing

        """
        from saplings.core.exceptions import InitializationError
        from saplings.core.validation import validate_required

        for param in self._required_params:
            try:
                validate_required(self._config_params.get(param), f"parameter '{param}'")
            except ValueError as e:
                raise InitializationError(
                    f"Missing required parameter '{param}' for Agent",
                    cause=e,
                )

    def _should_use_container(self) -> bool:
        """
        Determine whether to use the container based on the configuration.

        The container is used for advanced features like GASA, monitoring, and self-healing.
        For basic use cases, direct initialization is used for simplicity.

        Returns
        -------
            bool: True if container should be used, False otherwise

        """
        # Use container for advanced features
        return (
            self._config_params.get("enable_gasa", False)
            or self._config_params.get("enable_monitoring", True)
            or self._config_params.get("enable_self_healing", False)
            or self._config_params.get("enable_tool_factory", False)
            or
            # Other advanced features that require container
            self._config_params.get("planner_budget_strategy", "fixed") != "fixed"
        )

    # Add the methods to the class
    AgentBuilder.for_openai = for_openai
    AgentBuilder.for_anthropic = for_anthropic
    AgentBuilder.for_vllm = for_vllm
    AgentBuilder._validate_config = _validate_config
    AgentBuilder._should_use_container = _should_use_container

    return AgentBuilder
