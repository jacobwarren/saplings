from __future__ import annotations


def test_import_core_interfaces():
    """Test that we can import the core interfaces."""


def test_import_top_level():
    """Test that we can import from the top-level package."""


def test_import_agent_facade():
    """Test that we can import the agent facade."""
    from saplings.agent_facade import AgentFacade

    assert AgentFacade is not None

    from saplings.agent_facade_builder import AgentFacadeBuilder

    assert AgentFacadeBuilder is not None


def test_import_container():
    """Test that we can import the container modules."""
    from saplings.container import CircularDependencyError, LifecycleScope, SaplingsContainer, Scope

    assert SaplingsContainer is not None
    assert CircularDependencyError is not None
    assert LifecycleScope is not None
    assert Scope is not None

    from saplings.container_config import (
        configure_container,
        initialize_container,
        reset_container_config,
    )

    assert configure_container is not None
    assert initialize_container is not None
    assert reset_container_config is not None

    from saplings.container_hooks import (
        initialize_hooks,
        register_optimized_packers,
        register_vector_store,
    )

    assert initialize_hooks is not None
    assert register_optimized_packers is not None
    assert register_vector_store is not None

    from saplings.container_init import initialize_container, initialize_validators

    assert initialize_container is not None
    assert initialize_validators is not None

    from saplings.di import Container, container, inject, register, reset_container

    assert Container is not None
    assert container is not None
    assert inject is not None
    assert register is not None
    assert reset_container is not None
    from saplings import (
        IExecutionService,
        IMemoryManager,
        IModalityService,
        IModelCachingService,
        IModelInitializationService,
        IMonitoringService,
        IOrchestrationService,
        IPlannerService,
        IRetrievalService,
        ISelfHealingService,
        IToolService,
        IValidatorService,
    )

    assert IExecutionService is not None
    assert IMemoryManager is not None
    assert IModalityService is not None
    assert IModelInitializationService is not None
    assert IModelCachingService is not None
    assert IMonitoringService is not None
    assert IOrchestrationService is not None
    assert IPlannerService is not None
    assert IRetrievalService is not None
    assert ISelfHealingService is not None
    assert IToolService is not None
    assert IValidatorService is not None
    from saplings.api.core.interfaces import (
        IExecutionService,
        IMemoryManager,
        IModalityService,
        IModelCachingService,
        IModelInitializationService,
        IMonitoringService,
        IOrchestrationService,
        IPlannerService,
        IRetrievalService,
        ISelfHealingService,
        IToolService,
        IValidatorService,
    )

    assert IExecutionService is not None
    assert IMemoryManager is not None
    assert IModalityService is not None
    assert IModelInitializationService is not None
    assert IModelCachingService is not None
    assert IMonitoringService is not None
    assert IOrchestrationService is not None
    assert IPlannerService is not None
    assert IRetrievalService is not None
    assert ISelfHealingService is not None
    assert IToolService is not None
    assert IValidatorService is not None
