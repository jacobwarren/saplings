from __future__ import annotations

"""
Agent Builder build method for Saplings.

This module provides the build method for the AgentBuilder class.
"""

import logging
import warnings
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


def add_build_method(AgentBuilder):
    """Add the build method to the AgentBuilder class."""

    def build(self, use_container: Optional[bool] = None) -> "Agent":
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
            # Import here to avoid circular imports
            from saplings._internal.agent_class import Agent
            from saplings._internal.agent_module import AgentConfig
            from saplings.core.exceptions import InitializationError

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
                warnings.warn(
                    "The 'use_container' parameter is deprecated and will be removed in a future version. "
                    "The framework will automatically determine whether to use the container based on the configuration.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if use_container:
                # Import here to avoid circular imports
                from saplings._internal.container_config import initialize_container

                # Initialize the container with the config
                # This ensures all services are properly registered
                initialize_container(config)

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

    def _build_without_container(self, config: "AgentConfig") -> "Agent":
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
            from saplings._internal.agent_class import Agent
            from saplings.core.exceptions import InitializationError
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

        # Create monitoring service
        from saplings.services.monitoring_service import MonitoringService

        monitoring_service = MonitoringService(
            output_dir=config.output_dir, enabled=config.enable_monitoring
        )

        # Create a custom Agent class that doesn't use the container
        class CustomAgent(Agent):
            def __init__(self, config, **services):
                self.config = config
                # Set services directly
                for name, service in services.items():
                    setattr(self, f"_{name}", service)

                # Import here to avoid circular imports
                from saplings._internal.agent_facade import AgentFacade

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

    # Add the methods to the class
    AgentBuilder.build = build
    AgentBuilder._build_without_container = _build_without_container

    return AgentBuilder
