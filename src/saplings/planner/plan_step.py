"""
Plan step module for Saplings planner.

This module defines the PlanStep data structure and related enums.
"""

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PlanStepStatus(str, Enum):
    """Status of a plan step."""

    PENDING = "pending"  # Step has not been started
    IN_PROGRESS = "in_progress"  # Step is currently executing
    COMPLETED = "completed"  # Step has completed successfully
    FAILED = "failed"  # Step has failed
    SKIPPED = "skipped"  # Step was skipped (e.g., due to dependency failure)


class StepType(str, Enum):
    """Type of a plan step."""

    TASK = "task"  # General task
    RETRIEVAL = "retrieval"  # Information retrieval
    GENERATION = "generation"  # Content generation
    ANALYSIS = "analysis"  # Analysis or reasoning
    TOOL_USE = "tool_use"  # Using a tool
    DECISION = "decision"  # Making a decision
    VERIFICATION = "verification"  # Verifying results


class StepPriority(str, Enum):
    """Priority of a plan step."""

    LOW = "low"  # Low priority
    MEDIUM = "medium"  # Medium priority
    HIGH = "high"  # High priority
    CRITICAL = "critical"  # Critical priority


class PlanStep(BaseModel):
    """
    Plan step data structure.

    A plan step represents a single unit of work in a plan, with associated
    cost estimates, dependencies, and metadata.
    """

    # Note: We're not using frozen=True to allow for mutability in tests
    # In a real implementation, we might want to use immutable objects for thread safety

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the step")
    task_description: str = Field(..., description="Description of the task to perform")
    step_type: StepType = Field(StepType.TASK, description="Type of the step")
    priority: StepPriority = Field(StepPriority.MEDIUM, description="Priority of the step")
    estimated_cost: float = Field(0.0, description="Estimated cost of the step in USD")
    actual_cost: Optional[float] = Field(None, description="Actual cost of the step in USD")
    estimated_tokens: int = Field(0, description="Estimated number of tokens required")
    actual_tokens: Optional[int] = Field(None, description="Actual number of tokens used")
    dependencies: List[str] = Field(default_factory=list, description="IDs of steps this step depends on")
    status: PlanStepStatus = Field(PlanStepStatus.PENDING, description="Current status of the step")
    result: Optional[Any] = Field(None, description="Result of the step execution")
    error: Optional[str] = Field(None, description="Error message if the step failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("estimated_cost")
    def validate_estimated_cost(cls, v):
        """Validate that estimated cost is non-negative."""
        if v < 0:
            raise ValueError("Estimated cost must be non-negative")
        return v

    @field_validator("actual_cost")
    def validate_actual_cost(cls, v):
        """Validate that actual cost is non-negative."""
        if v is not None and v < 0:
            raise ValueError("Actual cost must be non-negative")
        return v

    @field_validator("estimated_tokens")
    def validate_estimated_tokens(cls, v):
        """Validate that estimated tokens is non-negative."""
        if v < 0:
            raise ValueError("Estimated tokens must be non-negative")
        return v

    @field_validator("actual_tokens")
    def validate_actual_tokens(cls, v):
        """Validate that actual tokens is non-negative."""
        if v is not None and v < 0:
            raise ValueError("Actual tokens must be non-negative")
        return v

    def is_complete(self) -> bool:
        """
        Check if the step is complete.

        Returns:
            bool: True if the step is complete, False otherwise
        """
        return self.status in (PlanStepStatus.COMPLETED, PlanStepStatus.FAILED, PlanStepStatus.SKIPPED)

    def is_successful(self) -> bool:
        """
        Check if the step completed successfully.

        Returns:
            bool: True if the step completed successfully, False otherwise
        """
        return self.status == PlanStepStatus.COMPLETED

    def is_failed(self) -> bool:
        """
        Check if the step failed.

        Returns:
            bool: True if the step failed, False otherwise
        """
        return self.status == PlanStepStatus.FAILED

    def is_skipped(self) -> bool:
        """
        Check if the step was skipped.

        Returns:
            bool: True if the step was skipped, False otherwise
        """
        return self.status == PlanStepStatus.SKIPPED

    def is_pending(self) -> bool:
        """
        Check if the step is pending.

        Returns:
            bool: True if the step is pending, False otherwise
        """
        return self.status == PlanStepStatus.PENDING

    def is_in_progress(self) -> bool:
        """
        Check if the step is in progress.

        Returns:
            bool: True if the step is in progress, False otherwise
        """
        return self.status == PlanStepStatus.IN_PROGRESS

    def get_cost_difference(self) -> Optional[float]:
        """
        Get the difference between actual and estimated cost.

        Returns:
            Optional[float]: Difference between actual and estimated cost, or None if actual cost is not available
        """
        if self.actual_cost is None:
            return None
        return self.actual_cost - self.estimated_cost

    def get_token_difference(self) -> Optional[int]:
        """
        Get the difference between actual and estimated tokens.

        Returns:
            Optional[int]: Difference between actual and estimated tokens, or None if actual tokens is not available
        """
        if self.actual_tokens is None:
            return None
        return self.actual_tokens - self.estimated_tokens

    def has_dependency(self, step_id: str) -> bool:
        """
        Check if this step depends on another step.

        Args:
            step_id: ID of the step to check

        Returns:
            bool: True if this step depends on the given step, False otherwise
        """
        return step_id in self.dependencies

    def add_dependency(self, step_id: str) -> None:
        """
        Add a dependency to this step.

        Args:
            step_id: ID of the step to depend on
        """
        if step_id not in self.dependencies:
            self.dependencies.append(step_id)

    def remove_dependency(self, step_id: str) -> None:
        """
        Remove a dependency from this step.

        Args:
            step_id: ID of the step to remove dependency on
        """
        if step_id in self.dependencies:
            self.dependencies.remove(step_id)

    def update_status(self, status: PlanStepStatus) -> None:
        """
        Update the status of this step.

        Args:
            status: New status
        """
        self.status = status

    def complete(self, result: Any, actual_cost: float, actual_tokens: int) -> None:
        """
        Mark this step as completed.

        Args:
            result: Result of the step execution
            actual_cost: Actual cost of the step in USD
            actual_tokens: Actual number of tokens used
        """
        self.status = PlanStepStatus.COMPLETED
        self.result = result
        self.actual_cost = actual_cost
        self.actual_tokens = actual_tokens

    def fail(self, error: str, actual_cost: Optional[float] = None, actual_tokens: Optional[int] = None) -> None:
        """
        Mark this step as failed.

        Args:
            error: Error message
            actual_cost: Actual cost of the step in USD (if available)
            actual_tokens: Actual number of tokens used (if available)
        """
        self.status = PlanStepStatus.FAILED
        self.error = error
        if actual_cost is not None:
            self.actual_cost = actual_cost
        if actual_tokens is not None:
            self.actual_tokens = actual_tokens

    def skip(self, reason: str) -> None:
        """
        Mark this step as skipped.

        Args:
            reason: Reason for skipping
        """
        self.status = PlanStepStatus.SKIPPED
        self.error = reason
        self.actual_cost = 0.0
        self.actual_tokens = 0

    def start(self) -> None:
        """Mark this step as in progress."""
        self.status = PlanStepStatus.IN_PROGRESS

    def reset(self) -> None:
        """Reset this step to pending status."""
        self.status = PlanStepStatus.PENDING
        self.result = None
        self.error = None
        self.actual_cost = None
        self.actual_tokens = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this step to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of this step
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """
        Create a step from a dictionary.

        Args:
            data: Dictionary representation of a step

        Returns:
            PlanStep: Created step
        """
        return cls(**data)
