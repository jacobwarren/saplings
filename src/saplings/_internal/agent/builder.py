from __future__ import annotations

"""
Agent Builder module for Saplings.

This module provides a builder class for creating Agent instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

from saplings.agent._internal.types import AgentBuilderProtocol

# Configure logging
logger = logging.getLogger(__name__)


class AgentBuilder(AgentBuilderProtocol):
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

    Example with container (default):
    ```python
    # Create a builder for Agent
    builder = AgentBuilder()

    # Configure the builder with dependencies and options
    agent = builder.with_provider("openai") \\
                  .with_model_name("gpt-4o") \\
                  .with_memory_path("./agent_memory") \\
                  .with_output_dir("./agent_output") \\
                  .with_gasa_enabled(True) \\
                  .with_monitoring_enabled(True) \\
                  .build()
    ```

    Example without container (simplified):
    ```python
    # Create a builder for Agent with simplified initialization
    builder = AgentBuilder()

    # Configure the builder with minimal options
    agent = builder.with_provider("openai") \\
                  .with_model_name("gpt-4o") \\
                  .with_tools([Calculator(), WebSearch()]) \\
                  .build(use_container=False)
    ```

    Example with configuration presets:
    ```python
    # Create an agent with minimal configuration
    agent = AgentBuilder.minimal("openai", "gpt-4o").build()

    # Create an agent with standard configuration
    agent = AgentBuilder.standard("openai", "gpt-4o").build()

    # Create an agent optimized for OpenAI
    agent = AgentBuilder.for_openai("gpt-4o").build()
    ```
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

    def with_memory_store(self, memory_store) -> AgentBuilder:
        """
        Set the memory store.

        Args:
        ----
            memory_store: Memory store instance

        Returns:
        -------
            The builder instance for method chaining

        """
        self._memory_store = memory_store
        return self

    def with_dependency_graph(self, dependency_graph) -> AgentBuilder:
        """
        Set the dependency graph.

        Args:
        ----
            dependency_graph: Dependency graph instance

        Returns:
        -------
            The builder instance for method chaining

        """
        self._dependency_graph = dependency_graph
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
        from saplings._internal.agent.config import AgentConfig

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
        from saplings._internal.agent.config import AgentConfig

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
        from saplings._internal.agent.config import AgentConfig

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

    def _validate_config(self) -> None:
        """
        Validate that all required configuration parameters are provided.

        Raises
        ------
            ValueError: If a required parameter is missing

        """
        for param in self._required_params:
            if self._config_params.get(param) is None:
                raise ValueError(f"Missing required parameter '{param}' for Agent")

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

    def build(self, use_container: Optional[bool] = None) -> Any:
        """
        Build the agent instance with the configured parameters.

        Args:
        ----
            use_container: DEPRECATED. Whether to use the container for dependency injection.
                           If None, will be determined automatically based on configuration.
                           This parameter will be removed in a future version.

        Returns:
        -------
            The initialized agent instance

        Raises:
        ------
            ValueError: If agent initialization fails

        """
        try:
            # Validate required parameters
            self._validate_config()

            # Import here to avoid circular imports
            from saplings._internal.agent.config import AgentConfig

            # Create the agent config
            config = AgentConfig(**self._config_params)

            # Determine whether to use container based on configuration
            if use_container is None:
                use_container = self._should_use_container()
                logger.debug(
                    f"Automatically determined use_container={use_container} based on configuration"
                )
            else:
                # Deprecation warning
                warnings.warn(
                    "The 'use_container' parameter is deprecated and will be removed in a future version. "
                    "The framework will automatically determine whether to use the container based on the configuration.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # Use a factory function to avoid circular imports
            def create_agent():
                # Import here to avoid circular imports
                from saplings._internal.agent.core import Agent
                from saplings._internal.container_config import initialize_container

                if use_container:
                    # Initialize the container with the config
                    # This ensures all services are properly registered
                    initialize_container(config)

                    # Create the agent instance
                    agent = Agent(config)
                    logger.info(
                        f"Built Agent with provider={config.provider}, model={config.model_name}"
                    )

                    # Set memory store and dependency graph if provided
                    if hasattr(self, "_memory_store"):
                        agent.memory_store = self._memory_store

                    if hasattr(self, "_dependency_graph"):
                        agent.dependency_graph = self._dependency_graph

                    return agent
                else:
                    # Create services directly without container
                    # This is a simplified initialization path for basic use cases

                    # TODO: Implement direct initialization without container
                    # For now, fall back to container-based initialization
                    logger.warning(
                        "Direct initialization without container is not yet implemented. Using container-based initialization."
                    )
                    initialize_container(config)
                    agent = Agent(config)

                    # Set memory store and dependency graph if provided
                    if hasattr(self, "_memory_store"):
                        agent.memory_store = self._memory_store

                    if hasattr(self, "_dependency_graph"):
                        agent.dependency_graph = self._dependency_graph

                    return agent

            return create_agent()

        except Exception as e:
            logger.error(f"Failed to build Agent: {e}")
            raise ValueError(f"Failed to build Agent: {e}") from e
