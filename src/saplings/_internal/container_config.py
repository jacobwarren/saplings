from __future__ import annotations

"""
Container configuration module for Saplings.

This module provides functions to configure the dependency injection container
with the services needed by the Saplings framework.
"""

import logging
from typing import Any

# Import the container directly from the internal module to avoid circular imports
from saplings.di._internal.container import container

logger = logging.getLogger(__name__)


def configure_monitoring_service(config: Any) -> None:
    """
    Configure the monitoring service in the container.

    This function registers the monitoring service with the container using the
    MonitoringServiceBuilder to create the service instance.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IMonitoringService
    from saplings.services._internal.builders.monitoring_service_builder import (
        MonitoringServiceBuilder,
    )

    # Register the monitoring service factory
    container.register(
        IMonitoringService,
        factory=lambda: MonitoringServiceBuilder()
        .with_output_dir(config.output_dir)
        .with_enabled(config.enable_monitoring)
        .build(),
        singleton=True,
    )

    logger.debug("Registered monitoring service with container")


def configure_validator_service(_config: Any = None) -> None:
    """
    Configure the validator service in the container.

    This function registers the validator service with the container using the
    ValidatorServiceBuilder to create the service instance with lazy initialization.

    Args:
    ----
        _config: The agent configuration (unused)

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IValidatorService
    from saplings.services._internal.builders.validator_service_builder import (
        ValidatorServiceBuilder,
    )

    # Register the validator service factory
    container.register(
        IValidatorService,
        factory=lambda: ValidatorServiceBuilder().build(),
        singleton=True,
    )

    logger.debug("Registered validator service with container (lazy initialization)")


def configure_model_initialization_service(config: Any) -> None:
    """
    Configure the model initialization service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IModelInitializationService
    from saplings.services._internal.builders.model_initialization_service_builder import (
        ModelInitializationServiceBuilder,
    )

    # Register the model initialization service factory
    container.register(
        IModelInitializationService,
        factory=lambda: ModelInitializationServiceBuilder()
        .with_provider(config.provider)
        .with_model_name(config.model_name)
        .with_model_parameters({"max_tokens": config.max_tokens, "temperature": config.temperature})
        .build(),
        singleton=True,
    )

    logger.debug("Registered model initialization service with container")


def configure_memory_manager_service(config: Any) -> None:
    """
    Configure the memory manager service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IMemoryManager
    from saplings.services._internal.builders.memory_manager_builder import MemoryManagerBuilder

    # Register the memory manager service factory
    container.register(
        IMemoryManager,
        factory=lambda: MemoryManagerBuilder().with_memory_path(config.memory_path).build(),
        singleton=True,
    )

    logger.debug("Registered memory manager service with container")


def configure_retrieval_service(config: Any) -> None:
    """
    Configure the retrieval service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IMemoryManager, IRetrievalService
    from saplings.services._internal.builders.retrieval_service_builder import (
        RetrievalServiceBuilder,
    )

    # Register the retrieval service factory with memory manager dependency
    def create_retrieval_service():
        # Resolve the memory manager from the container
        memory_manager = container.resolve(IMemoryManager)
        return (
            RetrievalServiceBuilder()
            .with_memory_store(memory_manager)
            .with_entropy_threshold(config.retrieval_entropy_threshold)
            .with_max_documents(config.retrieval_max_documents)
            .build()
        )

    container.register(
        IRetrievalService,
        factory=create_retrieval_service,
        singleton=True,
    )

    logger.debug("Registered retrieval service with container")


def configure_execution_service(config: Any) -> None:
    """
    Configure the execution service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IExecutionService
    from saplings.services._internal.builders.execution_service_builder import (
        ExecutionServiceBuilder,
    )

    # Register the execution service factory with lazy initialization
    def create_execution_service():
        # Create the execution service without model (lazy initialization)
        # The model will be provided when the service is actually used
        service = (
            ExecutionServiceBuilder()
            .with_config(config)  # Pass the full config instead of just a dict
            .build()
        )
        
        # The service will be initialized with the model when needed
        # through the async initialize() method
        return service

    container.register(IExecutionService, factory=create_execution_service, singleton=True)

    logger.debug("Registered execution service with container")


def configure_planner_service(config: Any) -> None:
    """
    Configure the planner service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IPlannerService
    from saplings.services._internal.builders.planner_service_builder import PlannerServiceBuilder

    # Register the planner service factory
    container.register(
        IPlannerService,
        factory=lambda: PlannerServiceBuilder()
        .with_budget_strategy(config.planner_budget_strategy)
        .with_total_budget(config.planner_total_budget)
        .with_allow_budget_overflow(config.planner_allow_budget_overflow)
        .build(),
        singleton=True,
    )

    logger.debug("Registered planner service with container")


def configure_tool_service(config: Any) -> None:
    """
    Configure the tool service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IToolService
    from saplings.services._internal.builders.tool_service_builder import ToolServiceBuilder

    # Register the tool service factory
    container.register(
        IToolService,
        factory=lambda: ToolServiceBuilder()
        .with_enabled(config.enable_tool_factory)
        .with_sandbox_enabled(config.tool_factory_sandbox_enabled)
        .with_allowed_imports(config.allowed_imports)
        .build(),
        singleton=True,
    )

    logger.debug("Registered tool service with container")


def configure_self_healing_service(config: Any) -> None:
    """
    Configure the self healing service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import ISelfHealingService
    from saplings.services._internal.builders.self_healing_service_builder import (
        SelfHealingServiceBuilder,
    )

    # Register the self healing service factory
    # Note: SelfHealingService requires patch_generator and success_pair_collector
    # For now, we'll create a minimal configuration
    container.register(
        ISelfHealingService,
        factory=lambda: SelfHealingServiceBuilder()
        .with_enabled(config.enable_self_healing)
        .build(),
        singleton=True,
    )

    logger.debug("Registered self healing service with container")


def configure_modality_service(config: Any) -> None:
    """
    Configure the modality service in the container.

    Args:
    ----
        config: The agent configuration

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IModalityService
    from saplings.services._internal.builders.modality_service_builder import ModalityServiceBuilder

    # Register the modality service factory
    container.register(
        IModalityService,
        factory=lambda: ModalityServiceBuilder()
        .with_supported_modalities(config.supported_modalities)
        .build(),
        singleton=True,
    )

    logger.debug("Registered modality service with container")


def configure_orchestration_service(_config: Any = None) -> None:
    """
    Configure the orchestration service in the container.

    Args:
    ----
        _config: The agent configuration (unused)

    """
    # Use lazy imports to avoid circular dependencies
    from saplings.api.core.interfaces import IOrchestrationService
    from saplings.services._internal.builders.orchestration_service_builder import (
        OrchestrationServiceBuilder,
    )

    # Register the orchestration service factory
    container.register(
        IOrchestrationService,
        factory=lambda: OrchestrationServiceBuilder().build(),
        singleton=True,
    )

    logger.debug("Registered orchestration service with container")


def configure_services(config: Any) -> None:
    """
    Configure all services in the container.

    This function registers all services needed by the Saplings framework with
    the container, using the appropriate builders to create service instances.

    Args:
    ----
        config: The agent configuration

    """
    # Configure all required services
    configure_monitoring_service(config)
    configure_validator_service(config)
    configure_model_initialization_service(config)
    configure_memory_manager_service(config)
    configure_retrieval_service(config)
    configure_execution_service(config)
    configure_planner_service(config)
    configure_tool_service(config)
    configure_self_healing_service(config)
    configure_modality_service(config)
    configure_orchestration_service(config)

    logger.debug("All services configured with container")
