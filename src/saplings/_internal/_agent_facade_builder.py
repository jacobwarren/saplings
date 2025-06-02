from __future__ import annotations

"""
Agent Facade Builder module for Saplings.

This module provides a builder class for creating AgentFacade instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from saplings._internal.agent_config import AgentConfig
from saplings._internal._agent_facade import AgentFacade
from saplings.core import InitializationError
from saplings.core._internal.builder import ServiceBuilder

logger = logging.getLogger(__name__)


class AgentFacadeBuilder(ServiceBuilder[AgentFacade]):
    """
    Builder for AgentFacade instances.

    This class provides a fluent interface for building AgentFacade instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for AgentFacade
    builder = AgentFacadeBuilder()

    # Configure the builder with dependencies and options
    facade = builder.with_config(config) \
                   .with_monitoring_service(monitoring_service) \
                   .with_model_service(model_service) \
                   .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the agent facade builder."""
        super().__init__(AgentFacade)
        self._config: Optional["AgentConfig"] = None
        self._testing: bool = False

    def with_config(self, config: "AgentConfig") -> "AgentFacadeBuilder":
        """
        Set the agent configuration.

        Args:
        ----
            config: Agent configuration

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config = config
        return self

    def with_testing(self, testing: bool) -> "AgentFacadeBuilder":
        """
        Set testing mode.

        Args:
        ----
            testing: Whether to enable testing mode

        Returns:
        -------
            The builder instance for method chaining

        """
        self._testing = testing
        return self

    def with_monitoring_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the monitoring service.

        Args:
        ----
            service: Monitoring service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("monitoring_service", service)
        return self

    def with_model_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the model service.

        Args:
        ----
            service: Model service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("model_service", service)
        return self

    def with_memory_manager(self, service) -> "AgentFacadeBuilder":
        """
        Set the memory manager.

        Args:
        ----
            service: Memory manager

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("memory_manager", service)
        return self

    def with_retrieval_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the retrieval service.

        Args:
        ----
            service: Retrieval service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("retrieval_service", service)
        return self

    def with_validator_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the validator service.

        Args:
        ----
            service: Validator service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("validator_service", service)
        return self

    def with_execution_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the execution service.

        Args:
        ----
            service: Execution service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("execution_service", service)
        return self

    def with_planner_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the planner service.

        Args:
        ----
            service: Planner service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("planner_service", service)
        return self

    def with_tool_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the tool service.

        Args:
        ----
            service: Tool service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("tool_service", service)
        return self

    def with_self_healing_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the self-healing service.

        Args:
        ----
            service: Self-healing service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("self_healing_service", service)
        return self

    def with_modality_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the modality service.

        Args:
        ----
            service: Modality service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("modality_service", service)
        return self

    def with_orchestration_service(self, service) -> "AgentFacadeBuilder":
        """
        Set the orchestration service.

        Args:
        ----
            service: Orchestration service

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("orchestration_service", service)
        return self

    def build(self) -> AgentFacade:
        """
        Build the agent facade instance with the configured dependencies.

        Returns
        -------
            The initialized agent facade instance

        Raises
        ------
            InitializationError: If agent facade initialization fails

        """
        try:
            # Validate required configuration
            if self._config is None:
                raise InitializationError("Agent configuration is required")

            # Add config and testing to dependencies
            self.with_dependency("config", self._config)
            self.with_dependency("testing", self._testing)

            # Build the agent facade
            return super().build()
        except Exception as e:
            if isinstance(e, InitializationError):
                raise
            raise InitializationError(
                f"Failed to initialize AgentFacade: {e}",
                cause=e,
            )
