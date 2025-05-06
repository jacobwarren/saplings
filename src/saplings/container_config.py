from __future__ import annotations

"""
Container configuration for Saplings dependency injection.

This module bootstraps the dependency injection container using the punq container.
It registers all services and provides transition functions from singletons.
"""


from typing import TYPE_CHECKING

from saplings.agent_config import AgentConfig
from saplings.core.function_registry import FunctionRegistry

# Core interfaces
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

# Core registries managed as DI singletons
from saplings.core.model_registry import ModelRegistry
from saplings.core.plugin import PluginRegistry
from saplings.di import container, reset_container
from saplings.executor import ExecutorConfig
from saplings.executor.config import RefinementStrategy
from saplings.gasa import GASAConfig
from saplings.gasa.config import FallbackStrategy, MaskStrategy
from saplings.integration.events import EventSystem
from saplings.memory.indexer import IndexerRegistry

# Core components
from saplings.planner import PlannerConfig
from saplings.planner.config import OptimizationStrategy
from saplings.retrieval import RetrievalConfig
from saplings.retrieval.config import EntropyConfig

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
from saplings.tools.browser_tools import BrowserManager
from saplings.validator.registry import ValidatorRegistry

if TYPE_CHECKING:
    from saplings.di import Container


def configure_container(config: AgentConfig | None = None) -> Container:
    """
    Configure the DI container with all services.

    Args:
    ----
        config: Optional agent configuration

    Returns:
    -------
        Configured container

    """
    # Reset container to clear any previous registrations
    reset_container()

    # Use provided config or create a default one
    if config is None:
        config = AgentConfig(provider="test", model_name="model")

    # Register configuration
    container.register(AgentConfig, instance=config)

    # Initialize optimized component hooks
    from saplings.container_hooks import initialize_hooks

    initialize_hooks(config)

    # Register core registries as singletons
    container.register(ModelRegistry, factory=ModelRegistry)

    # Create and register PluginRegistry instance to ensure it's available
    plugin_registry = PluginRegistry()
    container.register(PluginRegistry, instance=plugin_registry)

    container.register(FunctionRegistry, factory=FunctionRegistry)
    container.register(EventSystem, factory=EventSystem)
    container.register(IndexerRegistry, factory=IndexerRegistry)
    container.register(ValidatorRegistry, factory=ValidatorRegistry)

    # Register browser manager with the right instantiation pattern
    container.register(BrowserManager, factory=BrowserManager)

    # Register services

    # ModelService
    container.register(
        IModelService,
        factory=ModelService,
        provider=config.provider,
        model_name=config.model_name,
        **config.model_parameters,
    )

    # MonitoringService
    container.register(
        IMonitoringService,
        factory=MonitoringService,
        output_dir=config.output_dir,
        enabled=config.enable_monitoring,
    )

    # MemoryManager
    container.register(
        IMemoryManager,
        factory=lambda ms, config: MemoryManager(
            memory_path=config.memory_path,
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # RetrievalService
    container.register(
        IRetrievalService,
        factory=lambda mm, ms, config: RetrievalService(
            memory_store=mm.memory_store,
            config=RetrievalConfig(
                entropy=EntropyConfig(
                    threshold=config.retrieval_entropy_threshold,
                    max_documents=config.retrieval_max_documents,
                    max_iterations=3,
                    min_documents=5,
                    use_normalized_entropy=True,
                    window_size=3,
                )
            ),
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        mm=IMemoryManager,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # ValidatorService
    container.register(
        IValidatorService,
        factory=lambda model_svc, ms, config: ValidatorService(
            model=model_svc.get_model(),
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        model_svc=IModelService,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # PatchGenerator and SuccessPairCollector for SelfHealingService
    container.register(
        PatchGenerator,
        factory=lambda config: PatchGenerator(max_retries=config.self_healing_max_retries),
        config=AgentConfig,
    )

    container.register(
        SuccessPairCollector,
        factory=lambda config: SuccessPairCollector(
            output_dir=f"{config.output_dir}/success_pairs"
        ),
        config=AgentConfig,
    )

    # SelfHealingService
    container.register(
        ISelfHealingService,
        factory=lambda pg, spc, ms, config: SelfHealingService(
            patch_generator=pg,
            success_pair_collector=spc,
            enabled=config.enable_self_healing,
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        pg=PatchGenerator,
        spc=SuccessPairCollector,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # ExecutionService
    def create_gasa_config(config: AgentConfig) -> GASAConfig | None:
        if not config.enable_gasa:
            return None

        # Check if we're using a third-party LLM API
        is_third_party_api = config.provider in ["openai", "anthropic"]
        # Check if we're using vLLM
        is_vllm = config.provider == "vllm"

        if is_third_party_api:
            # For third-party APIs, use the prompt composer or shadow model
            gasa_config = GASAConfig.for_openai()

            # Apply agent config settings
            gasa_config.max_hops = config.gasa_max_hops
            gasa_config.mask_strategy = MaskStrategy(config.gasa_strategy)

            # Override fallback strategy if prompt composer is enabled
            fallback_strategy = config.gasa_fallback
            if fallback_strategy == "block_diagonal" and config.gasa_prompt_composer:
                fallback_strategy = "prompt_composer"
            gasa_config.fallback_strategy = FallbackStrategy(fallback_strategy)

            # Apply shadow model settings
            gasa_config.enable_shadow_model = config.gasa_shadow_model
            gasa_config.shadow_model_name = config.gasa_shadow_model_name

            # Apply prompt composer settings
            gasa_config.enable_prompt_composer = config.gasa_prompt_composer

            return gasa_config

        if is_vllm:
            # For vLLM, use the native tokenizer and standard GASA configuration
            gasa_config = GASAConfig.for_vllm()

            # Apply agent config settings
            gasa_config.max_hops = config.gasa_max_hops
            gasa_config.mask_strategy = MaskStrategy(config.gasa_strategy)
            gasa_config.fallback_strategy = FallbackStrategy(config.gasa_fallback)
            gasa_config.enable_prompt_composer = config.gasa_prompt_composer

            return gasa_config

        # For other local models, use the standard GASA configuration
        gasa_config = GASAConfig.for_local_models()

        # Apply agent config settings
        gasa_config.max_hops = config.gasa_max_hops
        gasa_config.mask_strategy = MaskStrategy(config.gasa_strategy)
        gasa_config.fallback_strategy = FallbackStrategy(config.gasa_fallback)
        gasa_config.enable_shadow_model = config.gasa_shadow_model
        gasa_config.shadow_model_name = config.gasa_shadow_model_name

        return gasa_config

    container.register(
        IExecutionService,
        factory=lambda model_svc, mm, ms, config: ExecutionService(
            model=model_svc.get_model(),
            config=ExecutorConfig(
                enable_gasa=config.enable_gasa,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                verification_strategy=config.executor_verification_strategy,
                execution_model_provider=None,
                execution_model_name=None,
                enable_speculative_execution=True,
                draft_temperature=0.2,
                final_temperature=0.7,
                max_draft_tokens=None,
                max_final_tokens=None,
                enable_streaming=True,
                stream_chunk_size=10,
                gasa_config=None,
                verification_threshold=0.7,
                refinement_strategy=RefinementStrategy.FEEDBACK,
                max_refinement_attempts=3,
                cache_results=True,
                cache_dir=None,
                log_level="INFO",
            ),
            gasa_config=create_gasa_config(config) if config.enable_gasa else None,
            dependency_graph=mm.dependency_graph if config.enable_gasa else None,
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        model_svc=IModelService,
        mm=IMemoryManager,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # PlannerService
    container.register(
        IPlannerService,
        factory=lambda model_svc, ms, config: PlannerService(
            model=model_svc.get_model(),
            config=PlannerConfig(
                budget_strategy=config.planner_budget_strategy,
                total_budget=config.planner_total_budget,
                allow_budget_overflow=config.planner_allow_budget_overflow,
                budget_overflow_margin=config.planner_budget_overflow_margin,
                optimization_strategy=OptimizationStrategy.BALANCED,
                max_steps=10,
                min_steps=1,
                enable_pruning=True,
                enable_parallelization=True,
                enable_caching=True,
                cache_dir=None,
            ),
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        model_svc=IModelService,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # ToolService
    container.register(
        IToolService,
        factory=lambda es, ms, config: ToolService(
            executor=es.executor,
            allowed_imports=config.allowed_imports,
            sandbox_enabled=config.tool_factory_sandbox_enabled,
            enabled=config.enable_tool_factory,
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        es=IExecutionService,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # ModalityService
    container.register(
        IModalityService,
        factory=lambda model_svc, ms, config: ModalityService(
            model=model_svc.get_model(),
            supported_modalities=config.supported_modalities,
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        model_svc=IModelService,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    # OrchestrationService
    container.register(
        IOrchestrationService,
        factory=lambda model_svc, ms, config: OrchestrationService(
            model=model_svc.get_model(),
            trace_manager=ms.trace_manager if config.enable_monitoring else None,
        ),
        model_svc=IModelService,
        ms=IMonitoringService,
        config=AgentConfig,
    )

    return container


def initialize_container(config: AgentConfig | None = None) -> None:
    """
    Initialize the container with default configuration.

    This is a convenience function for bootstrapping the container.

    Args:
    ----
        config: Optional agent configuration

    """
    configure_container(config)


# Initialize the container with default settings
# This ensures the container is ready to use when imported
initialize_container()
