"""
Saplings - A graphs-first, self-improving agent framework.

Saplings is a lightweight (≤ 1.2 k LOC core) framework for building domain-aware,
self-critiquing autonomous agents.

Key pillars:
1. Structural Memory — vector + graph store per corpus.
2. Cascaded, Entropy-Aware Retrieval — TF-IDF → embeddings → graph expansion.
3. Guard-railed Generation — Planner with budget, Executor with speculative draft/verify.
4. Judge & Validator Loop — reflexive scoring, self-healing patches.
5. Extensibility — hot-pluggable models, tools, validators.
6. Graph-Aligned Sparse Attention (GASA) — graph-conditioned attention masks for faster, better-grounded reasoning.
"""

__version__ = "0.1.0"

# Import core modules
from saplings.core import (
    LLM,
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
    ModelURI,
    Plugin,
    PluginRegistry,
    PluginType,
    discover_plugins,
)

# Import adapters if available
try:
    from saplings.adapters import (
        VLLMAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        HuggingFaceAdapter,
    )
except ImportError:
    pass

# Import memory modules
from saplings.memory import (
    Document,
    DocumentMetadata,
    DependencyGraph,
    MemoryConfig,
    MemoryStore,
    VectorStore,
)

# Import retrieval modules
from saplings.retrieval import (
    CascadeRetriever,
    EmbeddingRetriever,
    EntropyCalculator,
    GraphExpander,
    RetrievalConfig,
    TFIDFRetriever,
)

# Import GASA modules
from saplings.gasa import (
    BlockDiagonalPacker,
    GASAConfig,
    MaskBuilder,
    MaskFormat,
    MaskType,
    MaskVisualizer,
)

# Import planner modules
from saplings.planner import (
    BasePlanner,
    BudgetStrategy,
    OptimizationStrategy,
    PlanStep,
    PlanStepStatus,
    PlannerConfig,
    SequentialPlanner,
    StepPriority,
    StepType,
)

# Import executor modules
from saplings.executor import (
    Executor,
    ExecutionResult,
    ExecutorConfig,
    RefinementStrategy,
    VerificationStrategy,
)

# Import self-healing modules
from saplings.self_heal import (
    PatchGenerator,
    PatchResult,
    PatchStatus,
    Patch,
    SuccessPairCollector,
    LoRaTrainer,
    LoRaConfig,
    TrainingMetrics,
    AdapterManager,
    AdapterMetadata,
    AdapterPriority,
    Adapter,
)

# Import orchestration modules
from saplings.orchestration import (
    AgentNode,
    CommunicationChannel,
    GraphRunnerConfig,
    GraphRunner,
    NegotiationStrategy,
)

# Import tool factory modules
from saplings.tool_factory import (
    ToolSpecification,
    ToolFactoryConfig,
    ToolTemplate,
    SecurityLevel,
    ToolFactory,
)

# Import integration modules
from saplings.integration import (
    HotLoader,
    HotLoaderConfig,
    ToolLifecycleManager,
    IntegrationManager,
    EventSystem,
    EventType,
    Event,
    EventListener,
)

# Import high-level agent
from saplings.agent import Agent, AgentConfig

# Initialize plugins on import
discover_plugins()

__all__ = [
    # Version
    "__version__",

    # Core classes
    "LLM",
    "LLMResponse",
    "ModelCapability",
    "ModelMetadata",
    "ModelRole",
    "ModelURI",
    "Plugin",
    "PluginRegistry",
    "PluginType",
    "discover_plugins",

    # Adapter classes
    "VLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HuggingFaceAdapter",

    # Memory classes
    "Document",
    "DocumentMetadata",
    "DependencyGraph",
    "MemoryConfig",
    "MemoryStore",
    "VectorStore",

    # Retrieval classes
    "CascadeRetriever",
    "EmbeddingRetriever",
    "EntropyCalculator",
    "GraphExpander",
    "RetrievalConfig",
    "TFIDFRetriever",

    # GASA classes
    "BlockDiagonalPacker",
    "GASAConfig",
    "MaskBuilder",
    "MaskFormat",
    "MaskType",
    "MaskVisualizer",

    # Planner classes
    "BasePlanner",
    "BudgetStrategy",
    "OptimizationStrategy",
    "PlanStep",
    "PlanStepStatus",
    "PlannerConfig",
    "SequentialPlanner",
    "StepPriority",
    "StepType",

    # Executor classes
    "Executor",
    "ExecutionResult",
    "ExecutorConfig",
    "RefinementStrategy",
    "VerificationStrategy",

    # Self-healing classes
    "PatchGenerator",
    "PatchResult",
    "PatchStatus",
    "Patch",
    "SuccessPairCollector",
    "LoRaTrainer",
    "LoRaConfig",
    "TrainingMetrics",
    "AdapterManager",
    "AdapterMetadata",
    "AdapterPriority",
    "Adapter",

    # Orchestration classes
    "AgentNode",
    "CommunicationChannel",
    "GraphRunnerConfig",
    "GraphRunner",
    "NegotiationStrategy",

    # Tool factory classes
    "ToolSpecification",
    "ToolFactoryConfig",
    "ToolTemplate",
    "SecurityLevel",
    "ToolFactory",

    # Integration classes
    "HotLoader",
    "HotLoaderConfig",
    "ToolLifecycleManager",
    "IntegrationManager",
    "EventSystem",
    "EventType",
    "Event",
    "EventListener",

    # High-level agent
    "Agent",
    "AgentConfig",
]
