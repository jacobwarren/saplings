from __future__ import annotations

"""
Saplings Public API.

This module provides the stable, public API for the Saplings framework.
All public interfaces should be imported from this module.

The API is organized into the following categories:
- Agent: Core agent functionality
- Models: Model adapters and LLM interfaces
- Tools: Tool definitions and utilities
- Memory: Memory store and dependency graph
- Services: Service builders and interfaces
- GASA: Graph-Aligned Sparse Attention functionality
- Tool Factory: Dynamic tool creation and secure hot loading
- Self-Healing: Patch generation and adapter management
"""

# Import and re-export the public API components
from saplings.api.agent import Agent, AgentBuilder, AgentConfig, AgentFacade, AgentFacadeBuilder
from saplings.api.modality import (
    AudioHandler,
    ImageHandler,
    ModalityConfig,
    ModalityHandler,
    ModalityType,
    TextHandler,
    VideoHandler,
    get_handler_for_modality,
)
from saplings.api.orchestration import (
    AgentNode,
    CommunicationChannel,
    GraphRunner,
    GraphRunnerConfig,
    NegotiationStrategy,
    OrchestrationConfig,
)
from saplings.api.security import (
    RedactingFilter,
    Sanitizer,
    install_global_filter,
    install_import_hook,
    redact,
    sanitize,
)
from saplings.api.tokenizers import (
    SHADOW_MODEL_AVAILABLE,
    SimpleTokenizer,
    TokenizerFactory,
)

# Import shadow model tokenizer if available
if SHADOW_MODEL_AVAILABLE:
    from saplings.api.tokenizers import ShadowModelTokenizer
from saplings.api.di import (
    Container,
    configure_container,
    container,
    reset_container,
    reset_container_config,
)
from saplings.api.document_node import DocumentNode

# Document compatibility functions have been removed as they were only for backward compatibility
from saplings.api.memory import (
    DependencyGraph,
    DependencyGraphBuilder,
    MemoryConfig,
    MemoryStore,
    MemoryStoreBuilder,
)
from saplings.api.memory.document import Document, DocumentMetadata
from saplings.api.memory.indexer import Indexer, IndexerRegistry, SimpleIndexer, get_indexer
from saplings.api.models import (
    LLM,
    AnthropicAdapter,
    LLMBuilder,
    LLMResponse,
    ModelMetadata,
    OpenAIAdapter,
)
from saplings.api.validator import (
    ExecutionValidator,
    KeywordValidator,
    LengthValidator,
    RuntimeValidator,
    StaticValidator,
    ValidationResult,
    ValidationStatus,
    ValidationStrategy,
    Validator,
    ValidatorConfig,
    ValidatorRegistry,
    ValidatorType,
    get_validator_registry,
)
from saplings.api.vector_store import InMemoryVectorStore, VectorStore, get_vector_store


# Lazy import heavy adapters
def _get_heavy_model_adapter(name: str):
    """Get heavy model adapter using lazy import."""
    from saplings.api.models import __getattr__ as models_getattr

    return models_getattr(name)


from saplings.api.browser_tools import (
    ClickTool,
    ClosePopupsTool,
    GetPageTextTool,
    GoBackTool,
    GoToTool,
    ScrollTool,
    SearchTextTool,
    WaitTool,
    close_browser,
    get_browser_tools,
    initialize_browser,
    save_screenshot,
)

# Service interfaces and core types
from saplings.api.core import (
    # Execution
    ExecutionContext,
    ExecutionResult,
    # GASA
    GasaConfig,
    GenerationContext,
    IExecutionService,
    IGasaService,
    # Judge
    IJudgeService,
    # Memory
    IMemoryManager,
    # Modality
    IModalityService,
    # Model
    IModelCachingService,
    IModelInitializationService,
    # Monitoring
    IMonitoringService,
    # Orchestration
    IOrchestrationService,
    # Planning
    IPlannerService,
    # Retrieval
    IRetrievalService,
    # Self-healing
    ISelfHealingService,
    # Tools
    IToolService,
    # Validation
    IValidatorService,
    JudgeConfig,
    JudgeResult,
    MemoryConfig,
    MemoryResult,
    ModalityConfig,
    ModalityResult,
    ModelCachingConfig,
    ModelContext,
    ModelInitializationConfig,
    MonitoringConfig,
    MonitoringEvent,
    OrchestrationConfig,
    OrchestrationResult,
    PlannerConfig,
    PlanningResult,
    RetrievalConfig,
    RetrievalResult,
    SelfHealingConfig,
    SelfHealingResult,
    ToolConfig,
    ToolResult,
    ValidationConfig,
    ValidationContext,
    ValidationResult,
)

# GASA functionality
from saplings.api.gasa import (
    BlockDiagonalPacker,
    FallbackStrategy,
    GASAConfig,
    GASAConfigBuilder,
    GASAService,
    GASAServiceBuilder,
    GraphDistanceCalculator,
    MaskFormat,
    MaskStrategy,
    MaskType,
    MaskVisualizer,
    StandardMaskBuilder,
    TokenMapper,
    block_pack,
)
from saplings.api.judge import (
    CritiqueFormat,
    JudgeAgent,
    JudgeConfig,
    JudgeResult,
    Rubric,
    RubricItem,
    ScoringDimension,
)
from saplings.api.mcp_tools import (
    MCPClient,
    MCPTool,
)
from saplings.api.models import ModelCapability, ModelRole
from saplings.api.monitoring import (
    BlameEdge,
    BlameGraph,
    BlameNode,
    MonitoringConfig,
    TraceManager,
    TraceViewer,
)

# Registry and service locator - these will be moved to proper API modules
from saplings.api.registry import (
    IndexerRegistry,
    PluginRegistry,
    PluginType,
    RegistryContext,
    ServiceLocator,
)
from saplings.api.retrieval import (
    CascadeRetriever,
    EmbeddingRetriever,
    EntropyCalculator,
    FaissVectorStore,
    GraphExpander,
    RetrievalConfig,
    TFIDFRetriever,
)

# Self-Healing capabilities
from saplings.api.self_heal import (
    Adapter,
    AdapterManager,
    AdapterMetadata,
    AdapterPriority,
    LoRaConfig,
    LoRaTrainer,
    Patch,
    PatchGenerator,
    PatchResult,
    PatchStatus,
    RetryStrategy,
    SelfHealingConfig,
    SuccessPairCollector,
    TrainingMetrics,
)
from saplings.api.services import (
    # Service implementations
    ExecutionService,
    # Service builders
    ExecutionServiceBuilder,
    GASAConfigBuilder,
    GASAServiceBuilder,
    JudgeService,
    JudgeServiceBuilder,
    MemoryManager,
    MemoryManagerBuilder,
    ModalityService,
    ModalityServiceBuilder,
    ModelServiceBuilder,
    OrchestrationService,
    OrchestrationServiceBuilder,
    PlannerService,
    PlannerServiceBuilder,
    RetrievalService,
    RetrievalServiceBuilder,
    SelfHealingService,
    SelfHealingServiceBuilder,
    ToolService,
    ToolServiceBuilder,
    ValidatorService,
    ValidatorServiceBuilder,
)

# Tool Factory and Secure Hot Loading
from saplings.api.tool_factory import (
    CodeSigner,
    DockerSandbox,
    E2BSandbox,
    Sandbox,
    SandboxType,
    SecureHotLoader,
    SecureHotLoaderConfig,
    SecurityLevel,
    SignatureVerifier,
    SigningLevel,
    ToolFactory,
    ToolFactoryConfig,
    ToolSpecification,
    ToolTemplate,
    ToolValidator,
    ValidationResult,
    create_secure_hot_loader,
)
from saplings.api.tools import (
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    GoogleSearchTool,
    PythonInterpreterTool,
    SpeechToTextTool,
    Tool,
    ToolCollection,
    ToolRegistry,
    UserInputTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    get_all_default_tools,
    get_default_tool,
    get_registered_tools,
    is_browser_tools_available,
    is_mcp_available,
    register_tool,
    tool,
    validate_tool,
    validate_tool_attributes,
    validate_tool_parameters,
)

# Core utilities
from saplings.api.utils import (
    async_run_sync,
    count_tokens,
    get_model_sync,
    get_tokens_remaining,
    run_sync,
    split_text_by_tokens,
    truncate_text_tokens,
)
from saplings.api.version import __version__

# Core configuration and exceptions
from saplings.core import (
    Config,
    ConfigurationError,
    ConfigValue,
    ModelError,
    ProviderError,
    ResourceExhaustedError,
    SaplingsError,
)

# Define __all__ to control what is exported
__all__ = [
    # Agent
    "Agent",
    "AgentBuilder",
    "AgentConfig",
    "AgentFacade",
    "AgentFacadeBuilder",
    # Container and DI
    "Container",
    "container",
    "reset_container",
    "configure_container",
    "reset_container_config",
    # Validators
    "ExecutionValidator",
    "KeywordValidator",
    "LengthValidator",
    "RuntimeValidator",
    "StaticValidator",
    "ValidationResult",
    "ValidationStatus",
    "ValidationStrategy",
    "Validator",
    "ValidatorConfig",
    "ValidatorRegistry",
    "ValidatorType",
    "get_validator_registry",
    # Document
    "Document",
    "DocumentMetadata",
    "DocumentNode",
    # Memory
    "DependencyGraphBuilder",
    "MemoryStoreBuilder",
    "MemoryStore",
    "DependencyGraph",
    "MemoryConfig",
    "SimpleIndexer",
    "Indexer",
    "IndexerRegistry",
    "get_indexer",
    "VectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
    # Retrieval
    "CascadeRetriever",
    "EmbeddingRetriever",
    "EntropyCalculator",
    "FaissVectorStore",
    "GraphExpander",
    "RetrievalConfig",
    "TFIDFRetriever",
    # Monitoring
    "BlameEdge",
    "BlameGraph",
    "BlameNode",
    "MonitoringConfig",
    "TraceManager",
    "TraceViewer",
    # Judge
    "CritiqueFormat",
    "JudgeAgent",
    "JudgeConfig",
    "JudgeResult",
    "Rubric",
    "RubricItem",
    "ScoringDimension",
    # Modality
    "AudioHandler",
    "ImageHandler",
    "ModalityHandler",
    "ModalityConfig",
    "ModalityType",
    "TextHandler",
    "VideoHandler",
    "get_handler_for_modality",
    # Orchestration
    "AgentNode",
    "CommunicationChannel",
    "GraphRunner",
    "GraphRunnerConfig",
    "NegotiationStrategy",
    "OrchestrationConfig",
    # Security
    "RedactingFilter",
    "Sanitizer",
    "install_global_filter",
    "install_import_hook",
    "redact",
    "sanitize",
    # Tokenizers
    "SimpleTokenizer",
    "TokenizerFactory",
    "SHADOW_MODEL_AVAILABLE",
    # Models
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "LLM",
    "LLMBuilder",
    "LLMResponse",
    "ModelCapability",
    "ModelMetadata",
    "ModelRole",
    "OpenAIAdapter",
    "VLLMAdapter",
    # Services
    "ExecutionService",
    "ExecutionServiceBuilder",
    "GASAConfigBuilder",
    "GASAServiceBuilder",
    "JudgeService",
    "JudgeServiceBuilder",
    "MemoryManager",
    "MemoryManagerBuilder",
    "ModalityService",
    "ModalityServiceBuilder",
    "ModelServiceBuilder",
    "OrchestrationService",
    "OrchestrationServiceBuilder",
    "PlannerService",
    "PlannerServiceBuilder",
    "RetrievalService",
    "RetrievalServiceBuilder",
    "SelfHealingService",
    "SelfHealingServiceBuilder",
    "ToolService",
    "ToolServiceBuilder",
    "ValidatorService",
    "ValidatorServiceBuilder",
    # Tools
    "DuckDuckGoSearchTool",
    "FinalAnswerTool",
    "GoogleSearchTool",
    "PythonInterpreterTool",
    "SpeechToTextTool",
    "Tool",
    "ToolCollection",
    "ToolRegistry",
    "UserInputTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    "get_all_default_tools",
    "get_default_tool",
    "get_registered_tools",
    "is_browser_tools_available",
    "is_mcp_available",
    "register_tool",
    "tool",
    # Browser Tools
    "ClickTool",
    "ClosePopupsTool",
    "GetPageTextTool",
    "GoBackTool",
    "GoToTool",
    "ScrollTool",
    "SearchTextTool",
    "WaitTool",
    "close_browser",
    "get_browser_tools",
    "initialize_browser",
    "save_screenshot",
    # MCP Tools
    "MCPClient",
    "MCPTool",
    # Tool Validation
    "validate_tool",
    "validate_tool_attributes",
    "validate_tool_parameters",
    # Registry and service locator
    "PluginRegistry",
    "PluginType",
    "RegistryContext",
    "ServiceLocator",
    "IndexerRegistry",
    # GASA
    "BlockDiagonalPacker",
    "FallbackStrategy",
    "GASAConfig",
    "GASAConfigBuilder",
    "GASAService",
    "GASAServiceBuilder",
    "GraphDistanceCalculator",
    "MaskFormat",
    "MaskStrategy",
    "MaskType",
    "MaskVisualizer",
    "StandardMaskBuilder",
    "TokenMapper",
    "block_pack",
    # Tool Factory and Secure Hot Loading
    "CodeSigner",
    "DockerSandbox",
    "E2BSandbox",
    "Sandbox",
    "SandboxType",
    "SecureHotLoader",
    "SecureHotLoaderConfig",
    "SecurityLevel",
    "SignatureVerifier",
    "SigningLevel",
    "ToolFactory",
    "ToolFactoryConfig",
    "ToolSpecification",
    "ToolTemplate",
    "ToolValidator",
    "ValidationResult",
    "create_secure_hot_loader",
    # Self-Healing
    "Adapter",
    "AdapterManager",
    "AdapterMetadata",
    "AdapterPriority",
    "LoRaConfig",
    "LoRaTrainer",
    "Patch",
    "PatchGenerator",
    "PatchResult",
    "PatchStatus",
    "RetryStrategy",
    "SelfHealingConfig",
    "SuccessPairCollector",
    "TrainingMetrics",
    # Configuration and exceptions
    "Config",
    "ConfigValue",
    "ConfigurationError",
    "ModelError",
    "ProviderError",
    "ResourceExhaustedError",
    "SaplingsError",
    # Core interfaces and types
    # Execution
    "ExecutionContext",
    "ExecutionResult",
    "IExecutionService",
    # GASA
    "GasaConfig",
    "IGasaService",
    # Judge
    "IJudgeService",
    "JudgeConfig",
    "JudgeResult",
    # Memory
    "IMemoryManager",
    "MemoryConfig",
    "MemoryResult",
    # Modality
    "IModalityService",
    "ModalityConfig",
    "ModalityResult",
    # Model
    "IModelCachingService",
    "ModelCachingConfig",
    "IModelInitializationService",
    "ModelInitializationConfig",
    "ModelContext",
    "GenerationContext",
    # Monitoring
    "IMonitoringService",
    "MonitoringConfig",
    "MonitoringEvent",
    # Orchestration
    "IOrchestrationService",
    "OrchestrationConfig",
    "OrchestrationResult",
    # Planning
    "IPlannerService",
    "PlannerConfig",
    "PlanningResult",
    # Retrieval
    "IRetrievalService",
    "RetrievalConfig",
    "RetrievalResult",
    # Self-healing
    "ISelfHealingService",
    "SelfHealingConfig",
    "SelfHealingResult",
    # Tools
    "IToolService",
    "ToolConfig",
    "ToolResult",
    # Validation
    "IValidatorService",
    "ValidationConfig",
    "ValidationContext",
    "ValidationResult",
    # Utilities
    "async_run_sync",
    "count_tokens",
    "get_model_sync",
    "get_tokens_remaining",
    "run_sync",
    "split_text_by_tokens",
    "truncate_text_tokens",
    # Version
    "__version__",
]


# Lazy import heavy adapters
def __getattr__(name: str):
    """Lazy import heavy adapters using centralized system."""
    if name in ["HuggingFaceAdapter", "VLLMAdapter"]:
        return _get_heavy_model_adapter(name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Import optional components with better error handling
import logging

logger = logging.getLogger(__name__)

# Conditionally add ShadowModelTokenizer to __all__ if available
if SHADOW_MODEL_AVAILABLE:
    __all__.append("ShadowModelTokenizer")
    logger.debug("ShadowModelTokenizer added to __all__")

# Note: MCP tools and browser tools are already imported directly above
# They are included in __all__ and don't need conditional imports here
