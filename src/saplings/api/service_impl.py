from __future__ import annotations

"""
Service implementation API module for Saplings.

This module provides the public API for service implementations.
"""

# No type imports needed

# Import internal implementations
# Use relative imports to avoid importing from other components' internal modules
from saplings._internal.services.execution_service import ExecutionService as _ExecutionService
from saplings._internal.services.judge_service import JudgeService as _JudgeService
from saplings._internal.services.memory_manager import MemoryManager as _MemoryManager
from saplings._internal.services.modality_service import ModalityService as _ModalityService
from saplings._internal.services.orchestration_service import (
    OrchestrationService as _OrchestrationService,
)
from saplings._internal.services.planner_service import PlannerService as _PlannerService
from saplings._internal.services.retrieval_service import RetrievalService as _RetrievalService
from saplings._internal.services.self_healing_service import (
    SelfHealingService as _SelfHealingService,
)
from saplings._internal.services.tool_service import ToolService as _ToolService
from saplings._internal.services.validator_service import ValidatorService as _ValidatorService
from saplings.api.stability import beta, stable


# Base Service
@stable
class Service:
    """
    Base class for all services.

    This class provides common functionality for all services.
    """

    def __init__(self, **kwargs):
        """
        Initialize the service.

        Args:
        ----
            **kwargs: Additional configuration options

        """
        self.config = kwargs
        self._name = kwargs.get("name", self.__class__.__name__)

    @property
    def name(self) -> str:
        """
        Get the name of the service.

        Returns
        -------
            str: Name of the service

        """
        return self._name


# Execution Service
@stable
class ExecutionService(_ExecutionService):
    """
    Service for executing tasks.

    This service provides functionality for executing tasks, including
    handling context, tools, and validation.
    """


# Judge Service
@stable
class JudgeService(_JudgeService):
    """
    Service for judging outputs.

    This service provides functionality for judging outputs, including
    evaluating quality, correctness, and other criteria.
    """


# Memory Manager
@stable
class MemoryManager(_MemoryManager):
    """
    Service for managing memory.

    This service provides functionality for managing memory, including
    adding documents, retrieving documents, and managing the dependency graph.
    """


# Modality Service
@beta
class ModalityService(_ModalityService):
    """
    Service for handling different modalities.

    This service provides functionality for handling different modalities,
    including text, images, audio, and video.
    """


# Orchestration Service
@stable
class OrchestrationService(_OrchestrationService):
    """
    Service for orchestrating the agent workflow.

    This service provides functionality for orchestrating the agent workflow,
    including retrieval, planning, execution, and validation.
    """


# Planner Service
@stable
class PlannerService(_PlannerService):
    """
    Service for planning tasks.

    This service provides functionality for planning tasks, including
    breaking down tasks into steps and allocating budget.
    """


# Retrieval Service
@stable
class RetrievalService(_RetrievalService):
    """
    Service for retrieving documents.

    This service provides functionality for retrieving documents from memory,
    including semantic search and filtering.
    """


# Self-Healing Service
@beta
class SelfHealingService(_SelfHealingService):
    """
    Service for self-healing.

    This service provides functionality for self-healing, including
    detecting and fixing issues, and learning from past mistakes.
    """


# Tool Service
@stable
class ToolService(_ToolService):
    """
    Service for managing tools.

    This service provides functionality for managing tools, including
    registering tools, creating dynamic tools, and executing tools.
    """


# Validator Service
@stable
class ValidatorService(_ValidatorService):
    """
    Service for validating outputs.

    This service provides functionality for validating outputs, including
    checking for correctness, completeness, and other criteria.
    """
