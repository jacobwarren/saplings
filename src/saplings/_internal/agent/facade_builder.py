from __future__ import annotations

"""
Agent facade builder module for Saplings.

This module provides the AgentFacadeBuilder class for creating AgentFacade instances.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saplings._internal.agent.config import AgentConfig
    from saplings._internal.agent.facade import AgentFacade
    from saplings.api.core.interfaces import (
        IExecutionService,
        IMemoryManager,
        IModalityService,
        IModelInitializationService,
        IMonitoringService,
        IOrchestrationService,
        IPlannerService,
        IRetrievalService,
        ISelfHealingService,
        IToolService,
        IValidatorService,
    )

# Configure logging
logger = logging.getLogger(__name__)


class AgentFacadeBuilder:
    """
    Builder for AgentFacade instances.

    This class provides a fluent interface for building AgentFacade instances
    with proper configuration and dependency injection.
    """

    def __init__(self) -> None:
        """Initialize the builder with default values."""
        self._config = None
        self._monitoring_service = None
        self._model_service = None
        self._memory_manager = None
        self._retrieval_service = None
        self._validator_service = None
        self._execution_service = None
        self._planner_service = None
        self._tool_service = None
        self._self_healing_service = None
        self._modality_service = None
        self._orchestration_service = None

    def with_config(self, config: "AgentConfig") -> AgentFacadeBuilder:
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

    def with_monitoring_service(self, service: "IMonitoringService") -> AgentFacadeBuilder:
        """
        Set the monitoring service.

        Args:
        ----
            service: Monitoring service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._monitoring_service = service
        return self

    def with_model_service(self, service: "IModelInitializationService") -> AgentFacadeBuilder:
        """
        Set the model service.

        Args:
        ----
            service: Model service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._model_service = service
        return self

    def with_memory_manager(self, service: "IMemoryManager") -> AgentFacadeBuilder:
        """
        Set the memory manager.

        Args:
        ----
            service: Memory manager

        Returns:
        -------
            The builder instance for method chaining

        """
        self._memory_manager = service
        return self

    def with_retrieval_service(self, service: "IRetrievalService") -> AgentFacadeBuilder:
        """
        Set the retrieval service.

        Args:
        ----
            service: Retrieval service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._retrieval_service = service
        return self

    def with_validator_service(self, service: "IValidatorService") -> AgentFacadeBuilder:
        """
        Set the validator service.

        Args:
        ----
            service: Validator service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._validator_service = service
        return self

    def with_execution_service(self, service: "IExecutionService") -> AgentFacadeBuilder:
        """
        Set the execution service.

        Args:
        ----
            service: Execution service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._execution_service = service
        return self

    def with_planner_service(self, service: "IPlannerService") -> AgentFacadeBuilder:
        """
        Set the planner service.

        Args:
        ----
            service: Planner service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._planner_service = service
        return self

    def with_tool_service(self, service: "IToolService") -> AgentFacadeBuilder:
        """
        Set the tool service.

        Args:
        ----
            service: Tool service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._tool_service = service
        return self

    def with_self_healing_service(self, service: "ISelfHealingService") -> AgentFacadeBuilder:
        """
        Set the self-healing service.

        Args:
        ----
            service: Self-healing service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._self_healing_service = service
        return self

    def with_modality_service(self, service: "IModalityService") -> AgentFacadeBuilder:
        """
        Set the modality service.

        Args:
        ----
            service: Modality service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._modality_service = service
        return self

    def with_orchestration_service(self, service: "IOrchestrationService") -> AgentFacadeBuilder:
        """
        Set the orchestration service.

        Args:
        ----
            service: Orchestration service

        Returns:
        -------
            The builder instance for method chaining

        """
        self._orchestration_service = service
        return self

    def build(self) -> "AgentFacade":
        """
        Build the AgentFacade instance.

        Returns
        -------
            The initialized AgentFacade instance

        Raises
        ------
            ValueError: If any required service is missing

        """
        # Import here to avoid circular imports
        from saplings._internal.agent.facade import AgentFacade

        # Validate required services
        if not self._config:
            raise ValueError("Config is required")
        if not self._monitoring_service:
            raise ValueError("Monitoring service is required")
        if not self._model_service:
            raise ValueError("Model service is required")
        if not self._memory_manager:
            raise ValueError("Memory manager is required")
        if not self._retrieval_service:
            raise ValueError("Retrieval service is required")
        if not self._validator_service:
            raise ValueError("Validator service is required")
        if not self._execution_service:
            raise ValueError("Execution service is required")
        if not self._planner_service:
            raise ValueError("Planner service is required")
        if not self._tool_service:
            raise ValueError("Tool service is required")
        if not self._self_healing_service:
            raise ValueError("Self-healing service is required")
        if not self._modality_service:
            raise ValueError("Modality service is required")
        if not self._orchestration_service:
            raise ValueError("Orchestration service is required")

        # Create and return the facade
        return AgentFacade(
            config=self._config,
            monitoring_service=self._monitoring_service,
            model_service=self._model_service,
            memory_manager=self._memory_manager,
            retrieval_service=self._retrieval_service,
            validator_service=self._validator_service,
            execution_service=self._execution_service,
            planner_service=self._planner_service,
            tool_service=self._tool_service,
            self_healing_service=self._self_healing_service,
            modality_service=self._modality_service,
            orchestration_service=self._orchestration_service,
        )
