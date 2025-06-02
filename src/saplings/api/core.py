from __future__ import annotations

"""
Core API module for Saplings.

This module provides the public API for core interfaces and types.
"""

from abc import ABC

from saplings.api.stability import stable


# Base component class
@stable
class SaplingsComponent(ABC):
    """Base class for all Saplings components."""


# Import interfaces
from saplings.core._internal.interfaces import (
    # Execution
    IExecutionService as _IExecutionService,
)
from saplings.core._internal.interfaces import (
    # GASA
    IGasaService as _IGasaService,
)
from saplings.core._internal.interfaces import (
    # Judge
    IJudgeService as _IJudgeService,
)
from saplings.core._internal.interfaces import (
    # Memory
    IMemoryManager as _IMemoryManager,
)
from saplings.core._internal.interfaces import (
    # Modality
    IModalityService as _IModalityService,
)
from saplings.core._internal.interfaces import (
    # Model
    IModelCachingService as _IModelCachingService,
)
from saplings.core._internal.interfaces import (
    IModelInitializationService as _IModelInitializationService,
)
from saplings.core._internal.interfaces import (
    # Monitoring
    IMonitoringService as _IMonitoringService,
)
from saplings.core._internal.interfaces import (
    # Orchestration
    IOrchestrationService as _IOrchestrationService,
)
from saplings.core._internal.interfaces import (
    # Planning
    IPlannerService as _IPlannerService,
)
from saplings.core._internal.interfaces import (
    # Retrieval
    IRetrievalService as _IRetrievalService,
)
from saplings.core._internal.interfaces import (
    # Self-healing
    ISelfHealingService as _ISelfHealingService,
)
from saplings.core._internal.interfaces import (
    # Tools
    IToolService as _IToolService,
)
from saplings.core._internal.interfaces import (
    # Validation
    IValidatorService as _IValidatorService,
)

# Import configs and results from individual modules
from saplings.core._internal.interfaces.gasa import GasaConfig as _GasaConfig
from saplings.core._internal.interfaces.judge import (
    JudgeConfig as _JudgeConfig,
)
from saplings.core._internal.interfaces.judge import (
    JudgeResult as _JudgeResult,
)
from saplings.core._internal.interfaces.memory import (
    MemoryConfig as _MemoryConfig,
)
from saplings.core._internal.interfaces.memory import (
    MemoryResult as _MemoryResult,
)
from saplings.core._internal.interfaces.modality import (
    ModalityConfig as _ModalityConfig,
)
from saplings.core._internal.interfaces.modality import (
    ModalityResult as _ModalityResult,
)
from saplings.core._internal.interfaces.model_caching import (
    ModelCachingConfig as _ModelCachingConfig,
)
from saplings.core._internal.interfaces.model_initialization import (
    ModelInitializationConfig as _ModelInitializationConfig,
)
from saplings.core._internal.interfaces.monitoring import (
    MonitoringConfig as _MonitoringConfig,
)
from saplings.core._internal.interfaces.monitoring import (
    MonitoringEvent as _MonitoringEvent,
)
from saplings.core._internal.interfaces.orchestration import (
    OrchestrationConfig as _OrchestrationConfig,
)
from saplings.core._internal.interfaces.orchestration import (
    OrchestrationResult as _OrchestrationResult,
)
from saplings.core._internal.interfaces.planning import (
    PlannerConfig as _PlannerConfig,
)
from saplings.core._internal.interfaces.planning import (
    PlanningResult as _PlanningResult,
)
from saplings.core._internal.interfaces.retrieval import (
    RetrievalConfig as _RetrievalConfig,
)
from saplings.core._internal.interfaces.retrieval import (
    RetrievalResult as _RetrievalResult,
)
from saplings.core._internal.interfaces.self_healing import (
    SelfHealingConfig as _SelfHealingConfig,
)
from saplings.core._internal.interfaces.self_healing import (
    SelfHealingResult as _SelfHealingResult,
)
from saplings.core._internal.interfaces.tools import (
    ToolConfig as _ToolConfig,
)
from saplings.core._internal.interfaces.tools import (
    ToolResult as _ToolResult,
)
from saplings.core._internal.interfaces.validation import (
    ValidationConfig as _ValidationConfig,
)

# Import types
from saplings.core._internal.types import (
    ExecutionContext as _ExecutionContext,
)
from saplings.core._internal.types import (
    ExecutionResult as _ExecutionResult,
)
from saplings.core._internal.types import (
    GenerationContext as _GenerationContext,
)
from saplings.core._internal.types import (
    ModelContext as _ModelContext,
)
from saplings.core._internal.types import (
    ValidationContext as _ValidationContext,
)
from saplings.core._internal.types import (
    ValidationResult as _ValidationResult,
)


# Re-export with stability annotations
# Execution
@stable
class ExecutionContext(_ExecutionContext):
    """Standard context for execution operations."""


@stable
class ExecutionResult(_ExecutionResult):
    """Standard result for execution operations."""


@stable
class IExecutionService(_IExecutionService):
    """Interface for execution operations."""


# GASA
@stable
class GasaConfig(_GasaConfig):
    """Configuration for GASA operations."""


@stable
class IGasaService(_IGasaService):
    """Interface for GASA operations."""


# Judge
@stable
class JudgeConfig(_JudgeConfig):
    """Configuration for judge operations."""


@stable
class JudgeResult(_JudgeResult):
    """Result of a judge operation."""


@stable
class IJudgeService(_IJudgeService):
    """Interface for judge operations."""


# Memory
@stable
class MemoryConfig(_MemoryConfig):
    """Configuration for memory operations."""


@stable
class MemoryResult(_MemoryResult):
    """Result of a memory operation."""


@stable
class IMemoryManager(_IMemoryManager):
    """Interface for memory management operations."""


# Modality
@stable
class ModalityConfig(_ModalityConfig):
    """Configuration for modality operations."""


@stable
class ModalityResult(_ModalityResult):
    """Result of a modality operation."""


@stable
class IModalityService(_IModalityService):
    """Interface for modality operations."""


# Model
@stable
class ModelCachingConfig(_ModelCachingConfig):
    """Configuration for model caching operations."""


@stable
class IModelCachingService(_IModelCachingService):
    """Interface for model caching operations."""


@stable
class ModelInitializationConfig(_ModelInitializationConfig):
    """Configuration for model initialization operations."""


@stable
class IModelInitializationService(_IModelInitializationService):
    """Interface for model initialization operations."""


@stable
class ModelContext(_ModelContext):
    """Standard context for model operations."""


@stable
class GenerationContext(_GenerationContext):
    """Standard context for generation operations."""


# Monitoring
@stable
class MonitoringConfig(_MonitoringConfig):
    """Configuration for monitoring operations."""


@stable
class MonitoringEvent(_MonitoringEvent):
    """Event for monitoring operations."""


@stable
class IMonitoringService(_IMonitoringService):
    """Interface for monitoring operations."""


# Orchestration
@stable
class OrchestrationConfig(_OrchestrationConfig):
    """Configuration for orchestration operations."""


@stable
class OrchestrationResult(_OrchestrationResult):
    """Result of an orchestration operation."""


@stable
class IOrchestrationService(_IOrchestrationService):
    """Interface for orchestration operations."""


# Planning
@stable
class PlannerConfig(_PlannerConfig):
    """Configuration for planning operations."""


@stable
class PlanningResult(_PlanningResult):
    """Result of a planning operation."""


@stable
class IPlannerService(_IPlannerService):
    """Interface for planning operations."""


# Retrieval
@stable
class RetrievalConfig(_RetrievalConfig):
    """Configuration for retrieval operations."""


@stable
class RetrievalResult(_RetrievalResult):
    """Result of a retrieval operation."""


@stable
class IRetrievalService(_IRetrievalService):
    """Interface for retrieval operations."""


# Self-healing
@stable
class SelfHealingConfig(_SelfHealingConfig):
    """Configuration for self-healing operations."""


@stable
class SelfHealingResult(_SelfHealingResult):
    """Result of a self-healing operation."""


@stable
class ISelfHealingService(_ISelfHealingService):
    """Interface for self-healing operations."""


# Tools
@stable
class ToolConfig(_ToolConfig):
    """Configuration for tool operations."""


@stable
class ToolResult(_ToolResult):
    """Result of a tool operation."""


@stable
class IToolService(_IToolService):
    """Interface for tool operations."""


# Validation
@stable
class ValidationConfig(_ValidationConfig):
    """Configuration for validation operations."""


@stable
class ValidationContext(_ValidationContext):
    """Standard context for validation operations."""


@stable
class ValidationResult(_ValidationResult):
    """Standard result for validation operations."""


@stable
class IValidatorService(_IValidatorService):
    """Interface for validation operations."""


# Export all interfaces and types
__all__ = [
    # Base component
    "SaplingsComponent",
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
]
