from __future__ import annotations

"""
Core interfaces for the Saplings framework.

This package provides abstract base classes and type definitions
that define the contracts for services and components in the framework.
"""


from saplings.core.interfaces.execution import IExecutionService
from saplings.core.interfaces.gasa import IGASAService
from saplings.core.interfaces.memory import IMemoryManager
from saplings.core.interfaces.modality import IModalityService
from saplings.core.interfaces.model import IModelService
from saplings.core.interfaces.monitoring import IMonitoringService
from saplings.core.interfaces.orchestration import IOrchestrationService
from saplings.core.interfaces.planning import IPlannerService
from saplings.core.interfaces.retrieval import IRetrievalService
from saplings.core.interfaces.self_healing import ISelfHealingService
from saplings.core.interfaces.tools import IToolService
from saplings.core.interfaces.validation import IValidatorService

__all__ = [
    "IExecutionService",
    "IGASAService",
    "IMemoryManager",
    "IModalityService",
    "IModelService",
    "IMonitoringService",
    "IOrchestrationService",
    "IPlannerService",
    "IRetrievalService",
    "ISelfHealingService",
    "IToolService",
    "IValidatorService",
]
