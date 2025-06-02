from __future__ import annotations

"""
Plan module for Saplings planner.

This module defines the Plan data structure.
"""


from typing import Any, List

from pydantic import BaseModel, Field

from saplings.planner._internal.models.step import PlanStep


class Plan(BaseModel):
    """
    Plan data structure.

    A plan represents a collection of steps to accomplish a task.
    """

    steps: List[PlanStep] = Field(default_factory=list, description="Steps in the plan")
    task: str = Field("", description="Task description")
    total_estimated_cost: float = Field(0.0, description="Total estimated cost in USD")
    total_actual_cost: float | None = Field(None, description="Total actual cost in USD")
    total_estimated_tokens: int = Field(0, description="Total estimated tokens")
    total_actual_tokens: int | None = Field(None, description="Total actual tokens")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def add_step(self, step: PlanStep) -> None:
        """
        Add a step to the plan.

        Args:
        ----
            step: Step to add

        """
        self.steps.append(step)
        self.total_estimated_cost += step.estimated_cost
        self.total_estimated_tokens += step.estimated_tokens

    def get_step_by_id(self, step_id: str) -> PlanStep | None:
        """
        Get a step by its ID.

        Args:
        ----
            step_id: Step ID

        Returns:
        -------
            Optional[PlanStep]: Step with the given ID, or None if not found

        """
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def update_step(self, updated_step: PlanStep) -> None:
        """
        Update a step in the plan.

        Args:
        ----
            updated_step: Updated step

        """
        for i, step in enumerate(self.steps):
            if step.id == updated_step.id:
                # Update total cost and tokens
                if step.actual_cost is not None and updated_step.actual_cost is not None:
                    if self.total_actual_cost is None:
                        self.total_actual_cost = 0.0
                    self.total_actual_cost -= step.actual_cost
                    self.total_actual_cost += updated_step.actual_cost

                if step.actual_tokens is not None and updated_step.actual_tokens is not None:
                    if self.total_actual_tokens is None:
                        self.total_actual_tokens = 0
                    self.total_actual_tokens -= step.actual_tokens
                    self.total_actual_tokens += updated_step.actual_tokens

                # Replace the step
                self.steps[i] = updated_step
                return

    def is_complete(self) -> bool:
        """
        Check if the plan is complete.

        Returns
        -------
            bool: True if all steps are complete, False otherwise

        """
        return all(step.is_complete() for step in self.steps)

    def is_successful(self) -> bool:
        """
        Check if the plan completed successfully.

        Returns
        -------
            bool: True if all steps completed successfully, False otherwise

        """
        return all(step.is_successful() for step in self.steps)

    def get_next_step(self) -> PlanStep | None:
        """
        Get the next executable step.

        Returns
        -------
            Optional[PlanStep]: Next executable step, or None if all steps are complete

        """
        # Get all step IDs
        all_step_ids = {step.id for step in self.steps}

        # Find the first step that is pending and has all dependencies completed
        for step in self.steps:
            if step.is_pending():
                # Check if all dependencies are completed
                dependencies_completed = True
                for dep_id in step.dependencies:
                    # Skip if dependency doesn't exist (might be a bug)
                    if dep_id not in all_step_ids:
                        continue

                    # Get the dependency step
                    dep_step = self.get_step_by_id(dep_id)
                    if dep_step is None or not dep_step.is_successful():
                        dependencies_completed = False
                        break

                if dependencies_completed:
                    return step

        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this plan to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of this plan

        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        """
        Create a plan from a dictionary.

        Args:
        ----
            data: Dictionary representation of a plan

        Returns:
        -------
            Plan: Created plan

        """
        return cls(**data)
