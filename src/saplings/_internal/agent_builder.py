from __future__ import annotations

"""
Agent Builder module for Saplings.

This module provides a builder class for creating Agent instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import Any, Dict, List, Optional

from saplings.agent import Agent
from saplings.agent_config import AgentConfig
from saplings.api.container import configure_container
from saplings.core.exceptions import InitializationError
from saplings.core.validation import validate_required

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

    Example with container (default):
    ```python
    # Create a builder for Agent
    builder = AgentBuilder()

    # Configure the builder with dependencies and options
    agent = builder.with_provider("openai") \
                  .with_model_name("gpt-4o") \
                  .with_memory_path("./agent_memory") \
                  .with_output_dir("./agent_output") \
                  .with_gasa_enabled(True) \
                  .with_monitoring_enabled(True) \
                  .build()
    ```

    Example without container (simplified):
    ```python
    # Create a builder for Agent with simplified initialization
    builder = AgentBuilder()

    # Configure the builder with minimal options
    agent = builder.with_provider("openai") \
                  .with_model_name("gpt-4o") \
                  .with_tools([Calculator(), WebSearch()]) \
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

    def build(self, use_container: Optional[bool] = None) -> Agent:
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
            InitializationError: If agent initialization fails

        """
        try:
            # Validate required parameters
            self._validate_config()

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
                import warnings

                warnings.warn(
                    "The 'use_container' parameter is deprecated and will be removed in a future version. "
                    "The framework will automatically determine whether to use the container based on the configuration.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if use_container:
                # Initialize the container with the config
                # This ensures all services are properly registered
                configure_container(config)

                # Create the agent instance
                agent = Agent(config)
                logger.info(
                    f"Built Agent with provider={config.provider}, model={config.model_name}"
                )
                return agent
            else:
                # Create services directly without container
                return self._build_without_container(config)
        except Exception as e:
            if isinstance(e, InitializationError):
                raise
            raise InitializationError(
                f"Failed to initialize Agent: {e}",
                cause=e,
            )

    def _build_without_container(self, config: AgentConfig) -> Agent:
        """
        Build the agent instance without using the container.

        This method creates all required services directly and wires them together
        without using the dependency injection container, which simplifies the
        initialization flow for basic use cases.

        Args:
        ----
            config: The agent configuration

        Returns:
        -------
            The initialized agent instance

        """
        # Import services and builders here to avoid circular imports
        try:
            from saplings.services.builders import (
                ExecutionServiceBuilder,
                JudgeServiceBuilder,
                ValidatorServiceBuilder,
            )
            from saplings.services.builders.model_initialization_service_builder import (
                ModelInitializationServiceBuilder as ModelServiceBuilder,
            )
            from saplings.services.tool_service import ToolService
        except ImportError as e:
            raise InitializationError(
                f"Failed to import service builders: {e}",
                cause=e,
            )

        # Create model service
        model_service = (
            ModelServiceBuilder()
            .with_provider(config.provider)
            .with_model_name(config.model_name)
            .with_model_parameters(config.model_parameters)
            .build()
        )

        # Get the model
        # Initialize the model directly
        model_service._init_model()
        model = model_service.model

        # Ensure model is not None
        if model is None:
            raise InitializationError(
                "Failed to initialize model: model is None",
                cause=None,
            )

        # Create judge service
        judge_service = JudgeServiceBuilder().with_model(model).build()

        # Create a validator registry
        from saplings.validator.registry import ValidatorRegistry

        validator_registry = ValidatorRegistry()

        # Create validator service
        validator_service = (
            ValidatorServiceBuilder()
            .with_model(model)
            .with_judge_service(judge_service)
            .with_validator_registry(validator_registry)
            .build()
        )

        # Create execution service
        execution_service = (
            ExecutionServiceBuilder().with_model(model).with_validator(validator_service).build()
        )

        # Create tool service with tool factory disabled
        tool_service = ToolService(
            executor=execution_service,
            allowed_imports=config.allowed_imports,
            sandbox_enabled=False,
            enabled=False,  # Disable tool factory to avoid initialization issues
        )

        # Register tools if provided
        if config.tools:
            for tool in config.tools:
                tool_service.register_tool(tool)

        # Create monitoring service using the builder
        from saplings.api.monitoring import MonitoringServiceBuilder

        monitoring_service = (
            MonitoringServiceBuilder()
            .with_output_dir(config.output_dir)
            .with_enabled(config.enable_monitoring)
            .build()
        )

        # Create a custom Agent class that doesn't use the container
        class CustomAgent(Agent):
            def __init__(self, config, **services):
                self.config = config
                # Set services directly
                for name, service in services.items():
                    setattr(self, f"_{name}", service)

                # Import here to avoid circular imports
                from saplings.agent_facade import AgentFacade

                # Create the internal facade using the services
                self._facade = AgentFacade(config, **services)

                logger.info("CustomAgent initialized with direct service references")

        # Create agent with direct service references
        agent = CustomAgent(
            config,
            model_service=model_service,
            execution_service=execution_service,
            validator_service=validator_service,
            tool_service=tool_service,
            monitoring_service=monitoring_service,
        )

        logger.info(
            f"Built Agent with provider={config.provider}, model={config.model_name} (without container)"
        )
        return agent
