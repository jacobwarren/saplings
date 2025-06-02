from __future__ import annotations

"""
Public API for the Planner module.

This module provides the public API for the Planner module, which is responsible
for breaking down complex tasks into manageable steps while respecting budget
constraints and optimizing for efficiency.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from saplings.api.stability import stable


# Budget Strategy
@stable
class BudgetStrategy(str, Enum):
    """
    Strategy for budget allocation.

    This enum defines the strategies for allocating budget to plan steps.
    """

    EQUAL = "equal"
    PROPORTIONAL = "proportional"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"


# Optimization Strategy
@stable
class OptimizationStrategy(str, Enum):
    """
    Strategy for plan optimization.

    This enum defines the strategies for optimizing plans.
    """

    NONE = "none"
    COST = "cost"
    TIME = "time"
    QUALITY = "quality"
    BALANCED = "balanced"


# Step Priority
@stable
class StepPriority(str, Enum):
    """
    Priority of a plan step.

    This enum defines the priorities for plan steps.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Step Type
@stable
class StepType(str, Enum):
    """
    Type of a plan step.

    This enum defines the types of plan steps.
    """

    TASK = "task"
    DECISION = "decision"
    RESEARCH = "research"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    CUSTOM = "custom"


# Plan Step Status
@stable
class PlanStepStatus(str, Enum):
    """
    Status of a plan step.

    This enum defines the possible statuses for plan steps.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Plan Step
@stable
class PlanStep:
    """
    A step in a plan.

    This class represents a single step in a plan, with metadata about
    dependencies, estimated cost, and status.
    """

    def __init__(
        self,
        id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        estimated_cost: float = 0.0,
        estimated_tokens: int = 0,
        priority: StepPriority = StepPriority.MEDIUM,
        step_type: StepType = StepType.TASK,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a plan step.

        Args:
        ----
            id: Unique identifier for the step
            description: Description of the step
            dependencies: List of step IDs that this step depends on
            estimated_cost: Estimated cost of the step in USD
            estimated_tokens: Estimated number of tokens required for the step
            priority: Priority of the step
            step_type: Type of the step
            metadata: Additional metadata for the step

        """
        self.id = id
        self.description = description
        self.dependencies = dependencies or []
        self.estimated_cost = estimated_cost
        self.estimated_tokens = estimated_tokens
        self.priority = priority
        self.step_type = step_type
        self.metadata = metadata or {}
        self.status = PlanStepStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.actual_cost = 0.0
        self.actual_tokens = 0

    def is_complete(self) -> bool:
        """
        Check if the step is complete.

        Returns
        -------
            bool: True if the step is complete, False otherwise

        """
        return self.status in (
            PlanStepStatus.COMPLETED,
            PlanStepStatus.FAILED,
            PlanStepStatus.SKIPPED,
        )

    def is_successful(self) -> bool:
        """
        Check if the step completed successfully.

        Returns
        -------
            bool: True if the step completed successfully, False otherwise

        """
        return self.status == PlanStepStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the step to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the step

        """
        return {
            "id": self.id,
            "description": self.description,
            "dependencies": self.dependencies,
            "estimated_cost": self.estimated_cost,
            "estimated_tokens": self.estimated_tokens,
            "priority": self.priority,
            "step_type": self.step_type,
            "metadata": self.metadata,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "actual_cost": self.actual_cost,
            "actual_tokens": self.actual_tokens,
        }


# Planner Config
@stable
class PlannerConfig:
    """
    Configuration for planners.

    This class defines the configuration options for planners.
    """

    def __init__(
        self,
        max_steps: int = 10,
        min_steps: int = 1,
        total_budget: float = 1.0,
        allow_budget_overflow: bool = False,
        budget_overflow_margin: float = 0.1,
        budget_strategy: BudgetStrategy = BudgetStrategy.EQUAL,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        cost_heuristics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize planner configuration.

        Args:
        ----
            max_steps: Maximum number of steps in a plan
            min_steps: Minimum number of steps in a plan
            total_budget: Total budget for the plan in USD
            allow_budget_overflow: Whether to allow the plan to exceed the budget
            budget_overflow_margin: Maximum allowed budget overflow as a fraction of the total budget
            budget_strategy: Strategy for allocating budget to steps
            optimization_strategy: Strategy for optimizing the plan
            cost_heuristics: Heuristics for estimating costs

        """
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.total_budget = total_budget
        self.allow_budget_overflow = allow_budget_overflow
        self.budget_overflow_margin = budget_overflow_margin
        self.budget_strategy = budget_strategy
        self.optimization_strategy = optimization_strategy
        self.cost_heuristics = cost_heuristics or {
            "max_cost_per_step": total_budget / max_steps if max_steps > 0 else total_budget,
            "default_cost_per_token": 0.00002,
            "default_tokens_per_step": 1000,
        }

    @classmethod
    def default(cls) -> "PlannerConfig":
        """
        Create a default planner configuration.

        Returns
        -------
            PlannerConfig: Default planner configuration

        """
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the configuration

        """
        return {
            "max_steps": self.max_steps,
            "min_steps": self.min_steps,
            "total_budget": self.total_budget,
            "allow_budget_overflow": self.allow_budget_overflow,
            "budget_overflow_margin": self.budget_overflow_margin,
            "budget_strategy": self.budget_strategy,
            "optimization_strategy": self.optimization_strategy,
            "cost_heuristics": self.cost_heuristics,
        }


# Base Planner
@stable
class BasePlanner(ABC):
    """
    Abstract base class for planners.

    This class defines the interface that all planners must implement.
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        """
        Initialize the planner.

        Args:
        ----
            config: Planner configuration

        """
        self.config = config or PlannerConfig.default()
        self.steps: List[PlanStep] = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0

    @abstractmethod
    async def create_plan(self, task: str, **kwargs) -> List[PlanStep]:
        """
        Create a plan for a task.

        Args:
        ----
            task: Task description
            **kwargs: Additional arguments

        Returns:
        -------
            List[PlanStep]: List of plan steps

        """

    @abstractmethod
    async def optimize_plan(self, steps: List[PlanStep], **kwargs) -> List[PlanStep]:
        """
        Optimize a plan.

        Args:
        ----
            steps: List of plan steps
            **kwargs: Additional arguments

        Returns:
        -------
            List[PlanStep]: Optimized list of plan steps

        """

    @abstractmethod
    async def execute_plan(self, steps: List[PlanStep], **kwargs) -> Tuple[bool, Any]:
        """
        Execute a plan.

        Args:
        ----
            steps: List of plan steps
            **kwargs: Additional arguments

        Returns:
        -------
            Tuple[bool, Any]: Success flag and result

        """

    def validate_plan(self, steps: List[PlanStep]) -> bool:
        """
        Validate a plan.

        Args:
        ----
            steps: List of plan steps

        Returns:
        -------
            bool: True if the plan is valid, False otherwise

        """
        # Implementation would go here
        return True

    def get_execution_order(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """
        Get the execution order for a plan.

        Args:
        ----
            steps: List of plan steps

        Returns:
        -------
            List[List[PlanStep]]: List of batches of steps to execute in order

        """
        # Implementation would go here
        return []

    def estimate_cost(self, steps: List[PlanStep]) -> float:
        """
        Estimate the total cost of a plan.

        Args:
        ----
            steps: List of plan steps

        Returns:
        -------
            float: Estimated total cost in USD

        """
        return sum(step.estimated_cost for step in steps)

    def estimate_tokens(self, steps: List[PlanStep]) -> int:
        """
        Estimate the total number of tokens required for a plan.

        Args:
        ----
            steps: List of plan steps

        Returns:
        -------
            int: Estimated total number of tokens

        """
        return sum(step.estimated_tokens for step in steps)


# Sequential Planner
@stable
class SequentialPlanner(BasePlanner):
    """
    Planner that creates and executes steps sequentially.

    This planner creates a linear sequence of steps and executes them in order.
    """

    async def create_plan(self, task: str, **kwargs) -> List[PlanStep]:
        """
        Create a sequential plan for a task.

        Args:
        ----
            task: Task description
            **kwargs: Additional arguments

        Returns:
        -------
            List[PlanStep]: List of plan steps

        """
        # Implementation would go here
        return []

    async def optimize_plan(self, steps: List[PlanStep], **kwargs) -> List[PlanStep]:
        """
        Optimize a sequential plan.

        Args:
        ----
            steps: List of plan steps
            **kwargs: Additional arguments

        Returns:
        -------
            List[PlanStep]: Optimized list of plan steps

        """
        # Implementation would go here
        return steps

    async def execute_plan(self, steps: List[PlanStep], **kwargs) -> Tuple[bool, Any]:
        """
        Execute a sequential plan.

        Args:
        ----
            steps: List of plan steps
            **kwargs: Additional arguments

        Returns:
        -------
            Tuple[bool, Any]: Success flag and result

        """
        # Implementation would go here
        return True, None


__all__ = [
    "BasePlanner",
    "BudgetStrategy",
    "OptimizationStrategy",
    "PlanStep",
    "PlanStepStatus",
    "PlannerConfig",
    "SequentialPlanner",
    "StepPriority",
    "StepType",
]
