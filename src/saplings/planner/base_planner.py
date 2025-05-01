"""
Base planner module for Saplings.

This module defines the BasePlanner abstract class that all planners must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from saplings.core.model_adapter import LLM, ModelRole
from saplings.monitoring.trace import TraceManager
from saplings.planner.config import PlannerConfig
from saplings.planner.plan_step import PlanStep, PlanStepStatus

logger = logging.getLogger(__name__)


class BasePlanner(ABC):
    """
    Abstract base class for planners.

    This class defines the interface that all planners must implement.
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        model: Optional[LLM] = None,
        trace_manager: Optional["TraceManager"] = None,
    ):
        """
        Initialize the planner.

        Args:
            config: Planner configuration
            model: LLM model to use for planning
            trace_manager: TraceManager for tracing execution
        """
        self.config = config or PlannerConfig.default()
        self.model = model
        self.steps: List[PlanStep] = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
        self.trace_manager = trace_manager
        self.tools = {}

        # Validate model if provided
        if self.model is not None:
            self._validate_model()

    def _validate_model(self) -> None:
        """
        Validate that the model is suitable for planning.

        Raises:
            ValueError: If the model is not suitable for planning
        """
        metadata = self.model.get_metadata()
        if ModelRole.PLANNER not in metadata.roles and ModelRole.GENERAL not in metadata.roles:
            raise ValueError(
                f"Model {metadata.name} is not suitable for planning. "
                f"It must have either PLANNER or GENERAL role."
            )

    @abstractmethod
    async def create_plan(self, task: str, **kwargs) -> List[PlanStep]:
        """
        Create a plan for a task.

        Args:
            task: Task description
            **kwargs: Additional arguments

        Returns:
            List[PlanStep]: List of plan steps
        """
        pass

    @abstractmethod
    async def optimize_plan(self, steps: List[PlanStep], **kwargs) -> List[PlanStep]:
        """
        Optimize a plan.

        Args:
            steps: List of plan steps
            **kwargs: Additional arguments

        Returns:
            List[PlanStep]: Optimized list of plan steps
        """
        pass

    @abstractmethod
    async def execute_plan(self, steps: List[PlanStep], **kwargs) -> Tuple[bool, Any]:
        """
        Execute a plan.

        Args:
            steps: List of plan steps
            **kwargs: Additional arguments

        Returns:
            Tuple[bool, Any]: Success flag and result
        """
        pass

    def validate_plan(self, steps: List[PlanStep]) -> bool:
        """
        Validate a plan.

        Args:
            steps: List of plan steps

        Returns:
            bool: True if the plan is valid, False otherwise
        """
        # Check if the plan is empty
        if not steps:
            logger.warning("Plan is empty")
            return False

        # Check if the plan has too many steps
        if len(steps) > self.config.max_steps:
            logger.warning(f"Plan has too many steps: {len(steps)} > {self.config.max_steps}")
            return False

        # Check if the plan has too few steps
        if len(steps) < self.config.min_steps:
            logger.warning(f"Plan has too few steps: {len(steps)} < {self.config.min_steps}")
            return False

        # Check for circular dependencies
        if self._has_circular_dependencies(steps):
            logger.warning("Plan has circular dependencies")
            return False

        # Check for missing dependencies
        if self._has_missing_dependencies(steps):
            logger.warning("Plan has missing dependencies")
            return False

        # Check if the plan exceeds the budget
        total_estimated_cost = sum(step.estimated_cost for step in steps)
        if total_estimated_cost > self.config.total_budget:
            if not self.config.allow_budget_overflow:
                logger.warning(
                    f"Plan exceeds budget: {total_estimated_cost} > {self.config.total_budget}"
                )
                return False

            # Check if the plan exceeds the budget with margin
            max_budget = self.config.total_budget * (1 + self.config.budget_overflow_margin)
            if total_estimated_cost > max_budget:
                logger.warning(
                    f"Plan exceeds budget with margin: {total_estimated_cost} > {max_budget}"
                )
                return False
            # If we get here, the plan is within the budget + margin, so it's valid
            return True

        # Check for steps with excessive cost
        max_step_cost = self.config.cost_heuristics.max_cost_per_step
        for step in steps:
            if step.estimated_cost > max_step_cost:
                logger.warning(
                    f"Step {step.id} exceeds maximum cost: {step.estimated_cost} > {max_step_cost}"
                )
                return False

        # If we get here, the plan is valid
        return True

    def _has_circular_dependencies(self, steps: List[PlanStep]) -> bool:
        """
        Check if a plan has circular dependencies.

        Args:
            steps: List of plan steps

        Returns:
            bool: True if the plan has circular dependencies, False otherwise
        """
        # Create a mapping from step ID to step
        step_map = {step.id: step for step in steps}

        # Check for circular dependencies using DFS
        visited = set()
        path = set()

        def has_cycle(step_id: str) -> bool:
            if step_id in path:
                return True
            if step_id in visited:
                return False

            visited.add(step_id)
            path.add(step_id)

            step = step_map.get(step_id)
            if step is None:
                return False

            for dep_id in step.dependencies:
                if has_cycle(dep_id):
                    return True

            path.remove(step_id)
            return False

        for step in steps:
            if step.id not in visited:
                if has_cycle(step.id):
                    return True

        return False

    def _has_missing_dependencies(self, steps: List[PlanStep]) -> bool:
        """
        Check if a plan has missing dependencies.

        Args:
            steps: List of plan steps

        Returns:
            bool: True if the plan has missing dependencies, False otherwise
        """
        # Create a set of all step IDs
        step_ids = {step.id for step in steps}

        # Check for missing dependencies
        for step in steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    logger.warning(f"Step {step.id} depends on missing step {dep_id}")
                    return True

        return False

    def get_execution_order(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """
        Get the execution order for a plan.

        Args:
            steps: List of plan steps

        Returns:
            List[List[PlanStep]]: List of batches of steps to execute in order
        """
        # Create a mapping from step ID to step
        step_map = {step.id: step for step in steps}

        # Create a set of completed steps
        completed = set()

        # Create a list of batches
        batches = []

        # Keep track of remaining steps
        remaining = set(step.id for step in steps)

        # Process steps in batches
        while remaining:
            # Find steps that can be executed in parallel
            batch = []
            for step_id in list(remaining):
                step = step_map[step_id]
                if all(dep_id in completed for dep_id in step.dependencies):
                    batch.append(step)
                    remaining.remove(step_id)

            # If no steps can be executed, there must be a cycle
            if not batch:
                logger.warning("Cannot determine execution order due to circular dependencies")
                return []

            # Add the batch to the list of batches
            batches.append(batch)

            # Mark steps in the batch as completed
            for step in batch:
                completed.add(step.id)

        return batches

    def estimate_cost(self, steps: List[PlanStep]) -> float:
        """
        Estimate the total cost of a plan.

        Args:
            steps: List of plan steps

        Returns:
            float: Estimated total cost in USD
        """
        return sum(step.estimated_cost for step in steps)

    def estimate_tokens(self, steps: List[PlanStep]) -> int:
        """
        Estimate the total number of tokens required for a plan.

        Args:
            steps: List of plan steps

        Returns:
            int: Estimated total number of tokens
        """
        return sum(step.estimated_tokens for step in steps)

    def get_step_by_id(self, step_id: str) -> Optional[PlanStep]:
        """
        Get a step by its ID.

        Args:
            step_id: Step ID

        Returns:
            Optional[PlanStep]: Step with the given ID, or None if not found
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_steps_by_status(self, status: PlanStepStatus) -> List[PlanStep]:
        """
        Get steps with a given status.

        Args:
            status: Step status

        Returns:
            List[PlanStep]: Steps with the given status
        """
        return [step for step in self.steps if step.status == status]

    def get_completed_steps(self) -> List[PlanStep]:
        """
        Get completed steps.

        Returns:
            List[PlanStep]: Completed steps
        """
        return self.get_steps_by_status(PlanStepStatus.COMPLETED)

    def get_failed_steps(self) -> List[PlanStep]:
        """
        Get failed steps.

        Returns:
            List[PlanStep]: Failed steps
        """
        return self.get_steps_by_status(PlanStepStatus.FAILED)

    def get_pending_steps(self) -> List[PlanStep]:
        """
        Get pending steps.

        Returns:
            List[PlanStep]: Pending steps
        """
        return self.get_steps_by_status(PlanStepStatus.PENDING)

    def get_in_progress_steps(self) -> List[PlanStep]:
        """
        Get steps in progress.

        Returns:
            List[PlanStep]: Steps in progress
        """
        return self.get_steps_by_status(PlanStepStatus.IN_PROGRESS)

    def get_skipped_steps(self) -> List[PlanStep]:
        """
        Get skipped steps.

        Returns:
            List[PlanStep]: Skipped steps
        """
        return self.get_steps_by_status(PlanStepStatus.SKIPPED)

    def get_plan_status(self) -> Dict[str, Any]:
        """
        Get the status of the plan.

        Returns:
            Dict[str, Any]: Plan status
        """
        completed = self.get_completed_steps()
        failed = self.get_failed_steps()
        pending = self.get_pending_steps()
        in_progress = self.get_in_progress_steps()
        skipped = self.get_skipped_steps()

        total_steps = len(self.steps)
        completed_steps = len(completed)
        failed_steps = len(failed)
        pending_steps = len(pending)
        in_progress_steps = len(in_progress)
        skipped_steps = len(skipped)

        progress = completed_steps / total_steps if total_steps > 0 else 0.0

        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "pending_steps": pending_steps,
            "in_progress_steps": in_progress_steps,
            "skipped_steps": skipped_steps,
            "progress": progress,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "is_complete": self.is_plan_complete(),
            "is_successful": self.is_plan_successful(),
        }

    def is_plan_complete(self) -> bool:
        """
        Check if the plan is complete.

        Returns:
            bool: True if the plan is complete, False otherwise
        """
        return all(step.is_complete() for step in self.steps)

    def is_plan_successful(self) -> bool:
        """
        Check if the plan completed successfully.

        Returns:
            bool: True if the plan completed successfully, False otherwise
        """
        return self.is_plan_complete() and all(
            step.is_successful() or step.is_skipped() for step in self.steps
        )

    def reset_plan(self) -> None:
        """Reset the plan to its initial state."""
        for step in self.steps:
            step.reset()
        self.total_cost = 0.0
        self.total_tokens = 0

    def clear_plan(self) -> None:
        """Clear the plan."""
        self.steps = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the planner to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the planner
        """
        return {
            "config": self.config.model_dump(),
            "steps": [step.to_dict() for step in self.steps],
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasePlanner":
        """
        Create a planner from a dictionary.

        Args:
            data: Dictionary representation of a planner

        Returns:
            BasePlanner: Created planner
        """
        config = PlannerConfig(**data["config"])
        planner = cls(config=config)
        planner.steps = [PlanStep.from_dict(step_data) for step_data in data["steps"]]
        planner.total_cost = data["total_cost"]
        planner.total_tokens = data["total_tokens"]
        return planner
