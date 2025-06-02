from __future__ import annotations

"""
Agent facade builder module.

This module provides a builder for AgentFacade instances.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from saplings.agent._internal.types import AgentFacadeBuilderProtocol
from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.exceptions import InitializationError

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saplings.agent._internal.agent_config import AgentConfig


class AgentFacadeBuilder(ServiceBuilder[Any], AgentFacadeBuilderProtocol):
    """
    Builder for AgentFacade instances.

    This class provides a fluent interface for building AgentFacade instances with
    proper configuration and dependency injection.
    """

    def __init__(self) -> None:
        """Initialize the agent facade builder."""
        # Initialize with a placeholder class that will be replaced at build time
        super().__init__(object)
        self._config: Optional[Any] = None
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

    def with_monitoring_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_model_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_memory_manager(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_retrieval_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_validator_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_execution_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_planner_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_tool_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_self_healing_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_modality_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def with_orchestration_service(self, service: Any) -> "AgentFacadeBuilder":
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

    def build(self) -> Any:
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

            # Use a factory function to create the AgentFacade to avoid circular imports
            return self._create_facade(self._dependencies)
        except Exception as e:
            if isinstance(e, InitializationError):
                raise
            raise InitializationError(
                f"Failed to initialize AgentFacade: {e}",
                cause=e,
            )

    @staticmethod
    def _create_facade(dependencies: dict) -> Any:
        """
        Create an AgentFacade instance.

        This method uses a factory approach to avoid circular imports.

        Args:
        ----
            dependencies: Dependencies for the AgentFacade

        Returns:
        -------
            AgentFacade: The created facade instance

        """
        # Use dynamic import to avoid circular imports
        import importlib

        # Import the facade module dynamically
        facade_module = importlib.import_module("saplings.agent._internal.agent_facade")
        AgentFacade = facade_module.AgentFacade

        # Create the facade instance
        facade = AgentFacade(**dependencies)

        return facade
