from __future__ import annotations

"""
Container module for dependency injection in Saplings.

This module provides a dependency injection container that manages
the lifecycle and dependencies of all services used in the framework.
It uses the dependency-injector library to implement IoC (Inversion of Control)
principles and provides:

- Multiple lifecycle scopes (singleton, transient, scoped)
- Circular dependency detection
- Lazy singleton initialization
- Instance tracking for memory management
"""


import logging
import os
import threading
import weakref
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, cast

from dependency_injector import containers, providers

# Import the AgentConfig
from saplings.agent_config import AgentConfig

# Core interfaces
# Core components
from saplings.executor import ExecutorConfig
from saplings.gasa import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.planner import PlannerConfig
from saplings.retrieval import RetrievalConfig

# Self-healing components
from saplings.self_heal.patch_generator import PatchGenerator
from saplings.self_heal.success_pair_collector import SuccessPairCollector
from saplings.services.execution_service import ExecutionService

# Service implementations
from saplings.services.memory_manager import MemoryManager
from saplings.services.modality_service import ModalityService
from saplings.services.model_service import ModelService
from saplings.services.monitoring_service import MonitoringService
from saplings.services.orchestration_service import OrchestrationService
from saplings.services.planner_service import PlannerService
from saplings.services.retrieval_service import RetrievalService
from saplings.services.self_healing_service import SelfHealingService
from saplings.services.tool_service import ToolService
from saplings.services.validator_service import ValidatorService

if TYPE_CHECKING:
    from saplings.core.interfaces import (
        IExecutionService,
        IMemoryManager,
        IModalityService,
        IModelService,
        IMonitoringService,
        IOrchestrationService,
        IPlannerService,
        IRetrievalService,
        ISelfHealingService,
        IToolService,
        IValidatorService,
    )

# Import logging

logger = logging.getLogger(__name__)


class LifecycleScope(Enum):
    """
    Lifecycle scopes for container-managed dependencies.

    - SINGLETON: One instance per container, shared by all clients
    - SCOPED: One instance per scope (e.g., request, session)
    - TRANSIENT: New instance created on each request
    """

    SINGLETON = "singleton"
    SCOPED = "scoped"
    TRANSIENT = "transient"


class CircularDependencyError(Exception):
    """Exception raised when a circular dependency is detected."""

    def __init__(self, dependency_chain: list[str]) -> None:
        """Initialize with the dependency chain."""
        chain_str = " -> ".join(dependency_chain)
        message = f"Circular dependency detected: {chain_str}"
        super().__init__(message)
        self.dependency_chain = dependency_chain


class Scope:
    """
    Represents a dependency injection scope for scoped services.

    Scoped services are created once per scope and shared within that scope,
    but different scopes get different instances.
    """

    def __init__(self, container: "SaplingsContainer", parent_scope: "Scope | None" = None) -> None:
        """
        Initialize a new scope.

        Args:
        ----
            container: The DI container
            parent_scope: Parent scope if any

        """
        self._container = container
        self._parent_scope = parent_scope
        self._instances: dict[str, Any] = {}

    def get_or_create(self, service_name: str, factory: Callable[[], Any]) -> Any:
        """
        Get an existing instance or create a new one.

        Args:
        ----
            service_name: Name of the service
            factory: Factory function to create the service

        Returns:
        -------
            Service instance

        """
        if service_name in self._instances:
            return self._instances[service_name]

        # Check parent scope if it exists
        if self._parent_scope is not None:
            parent_instance = self._parent_scope.get(service_name)
            if parent_instance is not None:
                return parent_instance

        # Create new instance
        instance = factory()
        self._instances[service_name] = instance
        return instance

    def get(self, service_name: str) -> Any | None:
        """
        Get an existing instance if available.

        Args:
        ----
            service_name: Name of the service

        Returns:
        -------
            Service instance or None

        """
        return self._instances.get(service_name)

    def dispose(self):
        """Dispose of all resources in this scope."""
        for instance in self._instances.values():
            if hasattr(instance, "dispose") and callable(instance.dispose):
                try:
                    instance.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing {instance.__class__.__name__}: {e}")

        self._instances.clear()


class SaplingsContainer(containers.DeclarativeContainer):
    """
    Enhanced dependency injection container for Saplings.

    This container manages the lifecycle and dependencies of all services
    used in the framework. It follows the Inversion of Control principle
    by centralizing service creation and dependency management.

    Features:
    - Multiple lifecycle scopes (singleton, transient, scoped)
    - Circular dependency detection
    - Lazy singleton initialization
    - Instance tracking for memory management
    """

    # Configuration
    config = providers.Dependency(AgentConfig)

    # Wiring configuration
    wiring_config = containers.WiringConfiguration(packages=["saplings"])

    # Track dependencies for circular reference detection
    _dependency_stack: threading.local = threading.local()

    # Registry of all singletons for proper disposal
    _singletons: dict[str, weakref.ReferenceType] = {}

    # Scope management
    _current_scope: threading.local = threading.local()

    # Service factories

    # Initialize monitoring service first as other services may depend on it
    monitoring_service = providers.Factory(
        MonitoringService,
        output_dir=providers.Callable(lambda conf: conf.output_dir, conf=config),
        enabled=providers.Callable(lambda conf: conf.enable_monitoring, conf=config),
    )

    # Model service
    model_service = providers.Factory(
        ModelService,
        provider=providers.Callable(lambda conf: conf.provider, conf=config),
        model_name=providers.Callable(lambda conf: conf.model_name, conf=config),
        model_parameters=providers.Callable(lambda conf: conf.model_parameters, conf=config),
    )

    # Memory manager
    memory_manager = providers.Factory(
        MemoryManager,
        memory_path=providers.Callable(lambda conf: conf.memory_path, conf=config),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Retrieval service
    retrieval_service = providers.Factory(
        RetrievalService,
        memory_store=providers.Callable(
            lambda memory: memory.memory_store,
            memory=memory_manager,
        ),
        config=providers.Factory(
            RetrievalConfig,
        ),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Validator service
    validator_service = providers.Factory(
        ValidatorService,
        model=providers.Callable(
            lambda model_svc: model_svc.get_model(),
            model_svc=model_service,
        ),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Execution service
    execution_service = providers.Factory(
        ExecutionService,
        model=providers.Callable(
            lambda model_svc: model_svc.get_model(),
            model_svc=model_service,
        ),
        config=providers.Factory(
            ExecutorConfig,
        ),
        gasa_config=providers.Callable(
            lambda conf: _create_gasa_config(conf) if conf.enable_gasa else None,
            conf=config,
        ),
        dependency_graph=providers.Callable(
            lambda memory, conf: memory.dependency_graph if conf.enable_gasa else None,
            memory=memory_manager,
            conf=config,
        ),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Planner service
    planner_service = providers.Factory(
        PlannerService,
        model=providers.Callable(
            lambda model_svc: model_svc.get_model(),
            model_svc=model_service,
        ),
        config=providers.Factory(
            PlannerConfig,
        ),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Tool service
    tool_service = providers.Factory(
        ToolService,
        executor=providers.Callable(
            lambda execution_svc: execution_svc.executor,
            execution_svc=execution_service,
        ),
        allowed_imports=providers.Callable(lambda conf: conf.allowed_imports, conf=config),
        sandbox_enabled=providers.Callable(
            lambda conf: conf.tool_factory_sandbox_enabled, conf=config
        ),
        enabled=providers.Callable(lambda conf: conf.enable_tool_factory, conf=config),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Self-healing components
    patch_generator = providers.Factory(
        PatchGenerator,
        max_retries=providers.Callable(lambda conf: conf.self_healing_max_retries, conf=config),
    )

    success_pair_collector = providers.Factory(
        SuccessPairCollector,
        output_dir=providers.Callable(
            lambda conf: os.path.join(conf.output_dir, "success_pairs"),
            conf=config,
        ),
    )

    # Self-healing service
    self_healing_service = providers.Factory(
        SelfHealingService,
        patch_generator=patch_generator,
        success_pair_collector=success_pair_collector,
        enabled=providers.Callable(lambda conf: conf.enable_self_healing, conf=config),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Modality service
    modality_service = providers.Factory(
        ModalityService,
        model=providers.Callable(
            lambda model_svc: model_svc.get_model(),
            model_svc=model_service,
        ),
        supported_modalities=providers.Callable(
            lambda conf: conf.supported_modalities, conf=config
        ),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    # Orchestration service
    orchestration_service = providers.Factory(
        OrchestrationService,
        model=providers.Callable(
            lambda model_svc: model_svc.get_model(),
            model_svc=model_service,
        ),
        trace_manager=providers.Callable(
            lambda monitoring: monitoring.trace_manager if monitoring.enabled else None,
            monitoring=monitoring_service,
        ),
    )

    def __init__(self) -> None:
        """Initialize container."""
        super().__init__()
        # Initialize threading locals
        if not hasattr(self.__class__._dependency_stack, "stack"):
            self.__class__._dependency_stack.stack = []

        if not hasattr(self.__class__._current_scope, "scope"):
            self.__class__._current_scope.scope = None

    def create_scope(self, parent_scope: "Scope | None" = None) -> Scope:
        """
        Create a new dependency injection scope.

        Args:
        ----
            parent_scope: Optional parent scope

        Returns:
        -------
            New scope

        """
        return Scope(self, parent_scope)

    def enter_scope(self, scope: "Scope | None" = None) -> Scope:
        """
        Enter a dependency injection scope.

        Args:
        ----
            scope: Scope to enter or None to create a new one

        Returns:
        -------
            The entered scope

        """
        if scope is None:
            scope = self.create_scope()

        self.__class__._current_scope.scope = scope
        return scope

    def exit_scope(self):
        """Exit the current scope."""
        if (
            hasattr(self.__class__._current_scope, "scope")
            and self.__class__._current_scope.scope is not None
        ):
            self.__class__._current_scope.scope.dispose()
            self.__class__._current_scope.scope = None

    def get_service(
        self, service_provider: providers.Provider, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> Any:
        """
        Get a service instance with specified lifecycle scope.

        Args:
        ----
            service_provider: Provider for the service
            scope: Lifecycle scope

        Returns:
        -------
            Service instance

        Raises:
        ------
            CircularDependencyError: If a circular dependency is detected

        """
        service_name = str(service_provider)

        # Check for circular dependencies
        if hasattr(self.__class__._dependency_stack, "stack"):
            dependency_stack = self.__class__._dependency_stack.stack

            if service_name in dependency_stack:
                # Circular dependency detected
                circular_chain = dependency_stack[dependency_stack.index(service_name) :] + [
                    service_name
                ]
                raise CircularDependencyError(circular_chain)

            # Add service to dependency stack
            dependency_stack.append(service_name)

        try:
            # Create or get the service based on scope
            if scope == LifecycleScope.SINGLETON:
                # Check if singleton already exists
                if service_name in self.__class__._singletons:
                    singleton_ref = self.__class__._singletons[service_name]
                    singleton = singleton_ref()
                    if singleton is not None:
                        return singleton

                # Create new singleton
                instance = service_provider()
                self.__class__._singletons[service_name] = weakref.ref(instance)
                return instance

            if scope == LifecycleScope.SCOPED:
                # Use current scope or create one if none exists
                current_scope = getattr(self.__class__._current_scope, "scope", None)
                if current_scope is None:
                    current_scope = self.enter_scope()

                # Get or create service in current scope
                return current_scope.get_or_create(service_name, service_provider)

            if scope == LifecycleScope.TRANSIENT:
                # Always create a new instance
                return service_provider()

            msg = f"Unknown scope: {scope}"
            raise ValueError(msg)

        finally:
            # Remove service from dependency stack
            if (
                hasattr(self.__class__._dependency_stack, "stack")
                and len(self.__class__._dependency_stack.stack) > 0
            ):
                self.__class__._dependency_stack.stack.pop()

    # Service accessors with lifecycle scopes

    def get_monitoring_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IMonitoringService:
        """Get the monitoring service."""
        return cast("IMonitoringService", self.get_service(self.monitoring_service, scope))

    def get_model_service(self, scope: LifecycleScope = LifecycleScope.SINGLETON) -> IModelService:
        """Get the model service."""
        return cast("IModelService", self.get_service(self.model_service, scope))

    def get_memory_manager(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IMemoryManager:
        """Get the memory manager."""
        return cast("IMemoryManager", self.get_service(self.memory_manager, scope))

    def get_retrieval_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IRetrievalService:
        """Get the retrieval service."""
        return cast("IRetrievalService", self.get_service(self.retrieval_service, scope))

    def get_validator_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IValidatorService:
        """Get the validator service."""
        return cast("IValidatorService", self.get_service(self.validator_service, scope))

    def get_execution_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IExecutionService:
        """Get the execution service."""
        return cast("IExecutionService", self.get_service(self.execution_service, scope))

    def get_planner_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IPlannerService:
        """Get the planner service."""
        return cast("IPlannerService", self.get_service(self.planner_service, scope))

    def get_tool_service(self, scope: LifecycleScope = LifecycleScope.SINGLETON) -> IToolService:
        """Get the tool service."""
        return cast("IToolService", self.get_service(self.tool_service, scope))

    def get_self_healing_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> ISelfHealingService:
        """Get the self-healing service."""
        return cast("ISelfHealingService", self.get_service(self.self_healing_service, scope))

    def get_modality_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IModalityService:
        """Get the modality service."""
        return cast("IModalityService", self.get_service(self.modality_service, scope))

    def get_orchestration_service(
        self, scope: LifecycleScope = LifecycleScope.SINGLETON
    ) -> IOrchestrationService:
        """Get the orchestration service."""
        return cast("IOrchestrationService", self.get_service(self.orchestration_service, scope))

    def dispose(self):
        """Dispose of all resources."""
        # Exit any active scopes
        self.exit_scope()

        # Dispose singletons with dispose method
        for service_name, singleton_ref in list(self.__class__._singletons.items()):
            singleton = singleton_ref()
            if (
                singleton is not None
                and hasattr(singleton, "dispose")
                and callable(singleton.dispose)
            ):
                try:
                    singleton.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing {service_name}: {e}")

        # Clear singleton registry
        self.__class__._singletons.clear()


def _create_gasa_config(config: AgentConfig) -> GASAConfig:
    """Create GASA configuration based on the agent config."""
    # Check if we're using a third-party LLM API
    is_third_party_api = config.provider in ["openai", "anthropic"]

    # Check if we're using vLLM
    is_vllm = config.provider == "vllm"

    # Create a base GASA config with all required parameters
    base_config = {
        "enabled": True,
        "max_hops": config.gasa_max_hops,
        "mask_strategy": MaskStrategy(config.gasa_strategy),
        "global_tokens": ["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
        "summary_token": "[SUM]",
        "add_summary_token": True,
        "block_size": 512,
        "overlap": 64,
        "soft_mask_temperature": 0.1,
        "cache_masks": True,
        "cache_dir": None,
        "visualize": False,
        "visualization_dir": None,
        "shadow_model_cache_dir": None,
        "core_tag": "[CORE_CTX]",
        "near_tag": "[NEAR_CTX]",
        "summary_tag": "[SUMMARY_CTX]",
    }

    # Configure GASA based on the model type
    if is_third_party_api:
        # For third-party APIs, use the prompt composer or shadow model
        fallback_strategy = config.gasa_fallback
        if fallback_strategy == "block_diagonal" and config.gasa_prompt_composer:
            # Override fallback strategy if prompt composer is enabled
            fallback_strategy = "prompt_composer"

        # For third-party APIs, we need a shadow model for tokenization
        gasa_config = GASAConfig(
            **base_config,
            fallback_strategy=FallbackStrategy(fallback_strategy),
            enable_shadow_model=config.gasa_shadow_model,
            shadow_model_name=config.gasa_shadow_model_name,
            shadow_model_device="cpu",
            enable_prompt_composer=config.gasa_prompt_composer,
            focus_tags=True,
        )
        logger.info(
            f"Using GASA with third-party LLM API optimizations (fallback: {fallback_strategy})"
        )
    elif is_vllm:
        # For vLLM, use the native tokenizer and standard GASA configuration
        # vLLM already has its own tokenizer, so we don't need a shadow model
        gasa_config = GASAConfig(
            **base_config,
            fallback_strategy=FallbackStrategy(config.gasa_fallback),
            enable_shadow_model=False,
            shadow_model_name="",
            shadow_model_device="cpu",
            enable_prompt_composer=config.gasa_prompt_composer,
            focus_tags=True,
        )
        logger.info("Using GASA with vLLM (using native tokenizer)")
    else:
        # For other local models, use the standard GASA configuration
        gasa_config = GASAConfig(
            **base_config,
            fallback_strategy=FallbackStrategy(config.gasa_fallback),
            enable_shadow_model=config.gasa_shadow_model,
            shadow_model_name=config.gasa_shadow_model_name,
            shadow_model_device="cpu",
            enable_prompt_composer=False,
            focus_tags=False,
        )
        logger.info(f"Using standard GASA configuration (strategy: {config.gasa_strategy})")

    return gasa_config
