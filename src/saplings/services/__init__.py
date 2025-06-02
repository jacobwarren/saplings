from __future__ import annotations

"""
Services module for Saplings.

This module re-exports the public API from saplings.api.services.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.services.

# Import interfaces from the public API
from saplings.api.core.interfaces import (
    IModelCachingService,
    IModelInitializationService,
    IMonitoringService,
)

__all__ = [
    # Service implementations
    "ExecutionService",
    "JudgeService",
    "MemoryManager",
    "ModalityService",
    "OrchestrationService",
    "PlannerService",
    "RetrievalService",
    "SelfHealingService",
    "ToolService",
    "ValidatorService",
    # Service interfaces
    "IModelInitializationService",
    "IModelCachingService",
    "IMonitoringService",
    # Service builders
    "ExecutionServiceBuilder",
    "GASAConfigBuilder",
    "GASAServiceBuilder",
    "JudgeServiceBuilder",
    "MemoryManagerBuilder",
    "ModalityServiceBuilder",
    "ModelServiceBuilder",
    "OrchestrationServiceBuilder",
    "PlannerServiceBuilder",
    "RetrievalServiceBuilder",
    "SelfHealingServiceBuilder",
    "ToolServiceBuilder",
    "ValidatorServiceBuilder",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__ and name not in (
        "IModelInitializationService",
        "IModelCachingService",
        "IMonitoringService",
    ):
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

        # Create a mapping of names to their values
        globals_dict = {
            # Service implementations
            "ExecutionService": ExecutionService,
            "JudgeService": JudgeService,
            "MemoryManager": MemoryManager,
            "ModalityService": ModalityService,
            "OrchestrationService": OrchestrationService,
            "PlannerService": PlannerService,
            "RetrievalService": RetrievalService,
            "SelfHealingService": SelfHealingService,
            "ToolService": ToolService,
            "ValidatorService": ValidatorService,
            # Service builders
            "ExecutionServiceBuilder": ExecutionServiceBuilder,
            "GASAConfigBuilder": GASAConfigBuilder,
            "GASAServiceBuilder": GASAServiceBuilder,
            "JudgeServiceBuilder": JudgeServiceBuilder,
            "MemoryManagerBuilder": MemoryManagerBuilder,
            "ModalityServiceBuilder": ModalityServiceBuilder,
            "ModelServiceBuilder": ModelServiceBuilder,
            "OrchestrationServiceBuilder": OrchestrationServiceBuilder,
            "PlannerServiceBuilder": PlannerServiceBuilder,
            "RetrievalServiceBuilder": RetrievalServiceBuilder,
            "SelfHealingServiceBuilder": SelfHealingServiceBuilder,
            "ToolServiceBuilder": ToolServiceBuilder,
            "ValidatorServiceBuilder": ValidatorServiceBuilder,
        }

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
