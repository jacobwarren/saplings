from __future__ import annotations

"""
Core interfaces for Saplings.

This module provides the public API for service interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from saplings.api.stability import stable


# Execution interfaces
@stable
@dataclass
class ExecutionContext:
    """Context for execution operations."""

    task_id: str
    inputs: Dict[str, Any]
    metadata: Dict[str, Any] = None


@stable
@dataclass
class ExecutionResult:
    """Result of execution operations."""

    task_id: str
    outputs: Dict[str, Any]
    status: str
    metadata: Dict[str, Any] = None


@stable
class IExecutionService(ABC):
    """Interface for execution operations."""

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute a task."""


# GASA interfaces
@stable
@dataclass
class GasaConfig:
    """Configuration for GASA operations."""

    enabled: bool = True
    max_hops: int = 2
    mask_format: str = "binary"
    mask_strategy: str = "standard"
    mask_type: str = "attention"
    fallback_strategy: str = "none"


@stable
class IGasaService(ABC):
    """Interface for GASA operations."""

    @abstractmethod
    def create_mask(self, graph: Any, tokens: List[str]) -> Any:
        """Create a mask from a graph and tokens."""


# Judge interfaces
@stable
@dataclass
class JudgeConfig:
    """Configuration for judge operations."""

    enabled: bool = True
    threshold: float = 0.7
    critique_format: str = "structured"
    include_scores: bool = True
    include_suggestions: bool = True


@stable
@dataclass
class JudgeResult:
    """Result of judge operations."""

    output_id: str
    scores: Dict[str, float]
    critique: str
    passed: bool
    metadata: Dict[str, Any] = None


@stable
class IJudgeService(ABC):
    """Interface for judge operations."""

    @abstractmethod
    def evaluate(self, output: str, criteria: Any) -> JudgeResult:
        """Evaluate an output against criteria."""


# Memory interfaces
@stable
@dataclass
class MemoryConfig:
    """Configuration for memory operations."""

    enabled: bool = True
    vector_store_type: str = "in_memory"
    enable_graph: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200


@stable
@dataclass
class MemoryResult:
    """Result of memory operations."""

    document_id: str
    status: str
    metadata: Dict[str, Any] = None


@stable
class IMemoryManager(ABC):
    """Interface for memory management operations."""

    @abstractmethod
    def add_document(self, document: Any) -> MemoryResult:
        """Add a document to memory."""


# Modality interfaces
@stable
@dataclass
class ModalityConfig:
    """Configuration for modality operations."""

    enabled: bool = True
    supported_modalities: List[str] = None


@stable
@dataclass
class ModalityResult:
    """Result of modality operations."""

    modality_type: str
    content: Any
    metadata: Dict[str, Any] = None


@stable
class IModalityService(ABC):
    """Interface for modality operations."""

    @abstractmethod
    def process(self, content: Any, modality_type: str) -> ModalityResult:
        """Process content of a specific modality."""


# Model interfaces
@stable
@dataclass
class ModelContext:
    """Context for model operations."""

    model_id: str
    parameters: Dict[str, Any] = None


@stable
@dataclass
class GenerationContext:
    """Context for generation operations."""

    prompt: str
    model_context: ModelContext
    parameters: Dict[str, Any] = None


@stable
@dataclass
class ModelCachingConfig:
    """Configuration for model caching operations."""

    enabled: bool = True
    cache_dir: str = None
    max_cache_size: int = None


@stable
@dataclass
class ModelInitializationConfig:
    """Configuration for model initialization operations."""

    enabled: bool = True
    model_dir: str = None
    use_gpu: bool = True


@stable
class IModelCachingService(ABC):
    """Interface for model caching operations."""

    @abstractmethod
    def get_cached_result(self, key: str) -> Any:
        """Get a cached result."""


@stable
class IModelInitializationService(ABC):
    """Interface for model initialization operations."""

    @abstractmethod
    def initialize_model(self, model_id: str) -> Any:
        """Initialize a model."""


# Monitoring interfaces
@stable
@dataclass
class MonitoringConfig:
    """Configuration for monitoring operations."""

    enabled: bool = True
    tracing_backend: str = "console"
    visualization_format: str = "html"
    visualization_output_dir: str = "./visualizations"


@stable
@dataclass
class MonitoringEvent:
    """Event for monitoring."""

    event_type: str
    data: Dict[str, Any]


@stable
class IMonitoringService(ABC):
    """Interface for monitoring operations."""

    @abstractmethod
    def log_event(self, event: MonitoringEvent) -> None:
        """Log an event."""


# Orchestration interfaces
@stable
@dataclass
class OrchestrationConfig:
    """Configuration for orchestration operations."""

    enabled: bool = True
    max_agents: int = 10
    negotiation_strategy: str = "consensus"


@stable
@dataclass
class OrchestrationResult:
    """Result of orchestration operations."""

    graph_id: str
    status: str
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = None


@stable
class IOrchestrationService(ABC):
    """Interface for orchestration operations."""

    @abstractmethod
    def run_graph(self, graph: Any, inputs: Dict[str, Any]) -> OrchestrationResult:
        """Run a graph with inputs."""


# Planning interfaces
@stable
@dataclass
class PlannerConfig:
    """Configuration for planning operations."""

    enabled: bool = True
    max_steps: int = 10
    planning_strategy: str = "sequential"


@stable
@dataclass
class PlanningResult:
    """Result of planning operations."""

    plan_id: str
    steps: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None


@stable
class IPlannerService(ABC):
    """Interface for planning operations."""

    @abstractmethod
    def create_plan(self, task: str, context: Dict[str, Any]) -> PlanningResult:
        """Create a plan for a task."""


# Retrieval interfaces
@stable
@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""

    enabled: bool = True
    top_k: int = 5
    similarity_threshold: float = 0.7
    retrieval_strategy: str = "hybrid"


@stable
@dataclass
class RetrievalResult:
    """Result of retrieval operations."""

    query: str
    documents: List[Any]
    metadata: Dict[str, Any] = None


@stable
class IRetrievalService(ABC):
    """Interface for retrieval operations."""

    @abstractmethod
    def retrieve(self, query: str, filters: Dict[str, Any] = None) -> RetrievalResult:
        """Retrieve documents for a query."""


# Self-healing interfaces
@stable
@dataclass
class SelfHealingConfig:
    """Configuration for self-healing operations."""

    enabled: bool = True
    retry_strategy: str = "exponential"
    max_retries: int = 3
    collect_success_pairs: bool = True


@stable
@dataclass
class SelfHealingResult:
    """Result of self-healing operations."""

    error_id: str
    fixed: bool
    patch: Any = None
    metadata: Dict[str, Any] = None


@stable
class ISelfHealingService(ABC):
    """Interface for self-healing operations."""

    @abstractmethod
    def fix_error(self, error: Any, context: Dict[str, Any]) -> SelfHealingResult:
        """Fix an error."""


# Tool interfaces
@stable
@dataclass
class ToolConfig:
    """Configuration for tool operations."""

    enabled: bool = True
    allow_dynamic_tools: bool = True
    max_execution_time: float = 30.0
    sandbox_execution: bool = True


@stable
@dataclass
class ToolResult:
    """Result of tool operations."""

    tool_id: str
    outputs: Any
    status: str
    metadata: Dict[str, Any] = None


@stable
class IToolService(ABC):
    """Interface for tool operations."""

    @abstractmethod
    def execute_tool(self, tool_id: str, inputs: Dict[str, Any]) -> ToolResult:
        """Execute a tool with inputs."""


# Validation interfaces
@stable
@dataclass
class ValidationConfig:
    """Configuration for validation operations."""

    enabled: bool = True
    fail_fast: bool = False
    parallel_validation: bool = True
    max_parallel_validators: int = 10


@stable
@dataclass
class ValidationContext:
    """Context for validation operations."""

    content: Any
    content_type: str
    metadata: Dict[str, Any] = None


@stable
@dataclass
class ValidationResult:
    """Result of validation operations."""

    valid: bool
    errors: List[Dict[str, Any]] = None
    warnings: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None


@stable
class IValidatorService(ABC):
    """Interface for validation operations."""

    @abstractmethod
    def validate(self, context: ValidationContext) -> ValidationResult:
        """Validate content."""


__all__ = [
    # Execution interfaces
    "ExecutionContext",
    "ExecutionResult",
    "IExecutionService",
    # GASA interfaces
    "GasaConfig",
    "IGasaService",
    # Judge interfaces
    "JudgeConfig",
    "JudgeResult",
    "IJudgeService",
    # Memory interfaces
    "MemoryConfig",
    "MemoryResult",
    "IMemoryManager",
    # Modality interfaces
    "ModalityConfig",
    "ModalityResult",
    "IModalityService",
    # Model interfaces
    "ModelContext",
    "GenerationContext",
    "ModelCachingConfig",
    "ModelInitializationConfig",
    "IModelCachingService",
    "IModelInitializationService",
    # Monitoring interfaces
    "MonitoringConfig",
    "MonitoringEvent",
    "IMonitoringService",
    # Orchestration interfaces
    "OrchestrationConfig",
    "OrchestrationResult",
    "IOrchestrationService",
    # Planning interfaces
    "PlannerConfig",
    "PlanningResult",
    "IPlannerService",
    # Retrieval interfaces
    "RetrievalConfig",
    "RetrievalResult",
    "IRetrievalService",
    # Self-healing interfaces
    "SelfHealingConfig",
    "SelfHealingResult",
    "ISelfHealingService",
    # Tool interfaces
    "ToolConfig",
    "ToolResult",
    "IToolService",
    # Validation interfaces
    "ValidationConfig",
    "ValidationContext",
    "ValidationResult",
    "IValidatorService",
]
