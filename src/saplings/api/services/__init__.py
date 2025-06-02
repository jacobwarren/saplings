from __future__ import annotations

"""
Services API module for Saplings.

This module provides the public API for service implementations.
"""

# Import from submodules
from saplings.api.services.base import Service
from saplings.api.services.execution import ExecutionService
from saplings.api.services.judge import JudgeService
from saplings.api.services.memory import MemoryManager
from saplings.api.services.modality import ModalityService
from saplings.api.services.orchestration import OrchestrationService
from saplings.api.services.planner import PlannerService
from saplings.api.services.retrieval import RetrievalService
from saplings.api.services.self_healing import SelfHealingService
from saplings.api.services.tool import ToolService
from saplings.api.services.validator import ValidatorService

# Define __all__ without the builders to avoid circular imports
__all__ = [
    # Service implementations
    "Service",
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
]


# Add builder imports and __all__ entries through a function to avoid circular imports
def _import_builders():
    """Import builders and add them to __all__."""
    # pylint: disable=global-statement
    global __all__

    # Import service builders
    from saplings.api.services.builders import (
        ExecutionServiceBuilder,
        GASAConfigBuilder,
        GASAServiceBuilder,
        JudgeServiceBuilder,
        MemoryManagerBuilder,
        ModalityServiceBuilder,
        ModelServiceBuilder,
        OrchestrationServiceBuilder,
        PlannerServiceBuilder,
        RetrievalServiceBuilder,
        SelfHealingServiceBuilder,
        ToolServiceBuilder,
        ValidatorServiceBuilder,
    )

    # Add builder names to __all__
    __all__ += [
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

    # Return the imported builders
    return {
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


# Use __getattr__ for lazy loading of builders
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in [
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
    ]:
        builders = _import_builders()
        return builders.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
