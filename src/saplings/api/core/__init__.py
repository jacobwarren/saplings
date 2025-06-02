from __future__ import annotations

"""
Core API module for Saplings.

This module provides the public API for core interfaces and types.
"""

# Import interfaces
from saplings.api.core.interfaces import (
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

__all__ = [
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
