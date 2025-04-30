"""
Tests for the plan step module.
"""

import pytest
from pydantic import ValidationError

from saplings.planner.plan_step import PlanStep, PlanStepStatus, StepPriority, StepType


class TestPlanStep:
    """Tests for the PlanStep class."""
    
    def test_init(self):
        """Test initialization."""
        step = PlanStep(
            task_description="Test task",
            step_type=StepType.TASK,
            priority=StepPriority.MEDIUM,
            estimated_cost=0.1,
            estimated_tokens=1000,
        )
        
        assert step.task_description == "Test task"
        assert step.step_type == StepType.TASK
        assert step.priority == StepPriority.MEDIUM
        assert step.estimated_cost == 0.1
        assert step.estimated_tokens == 1000
        assert step.dependencies == []
        assert step.status == PlanStepStatus.PENDING
        assert step.result is None
        assert step.error is None
        assert step.metadata == {}
    
    def test_validation(self):
        """Test validation."""
        # Test negative estimated cost
        with pytest.raises(ValidationError):
            PlanStep(
                task_description="Test task",
                estimated_cost=-0.1,
            )
        
        # Test negative estimated tokens
        with pytest.raises(ValidationError):
            PlanStep(
                task_description="Test task",
                estimated_tokens=-1000,
            )
        
        # Test negative actual cost
        with pytest.raises(ValidationError):
            PlanStep(
                task_description="Test task",
                actual_cost=-0.1,
            )
        
        # Test negative actual tokens
        with pytest.raises(ValidationError):
            PlanStep(
                task_description="Test task",
                actual_tokens=-1000,
            )
    
    def test_status_checks(self):
        """Test status check methods."""
        step = PlanStep(
            task_description="Test task",
        )
        
        # Initial status is PENDING
        assert step.is_pending()
        assert not step.is_complete()
        assert not step.is_successful()
        assert not step.is_failed()
        assert not step.is_skipped()
        assert not step.is_in_progress()
        
        # Update status to IN_PROGRESS
        step.update_status(PlanStepStatus.IN_PROGRESS)
        assert step.is_in_progress()
        assert not step.is_complete()
        assert not step.is_successful()
        assert not step.is_failed()
        assert not step.is_skipped()
        assert not step.is_pending()
        
        # Update status to COMPLETED
        step.update_status(PlanStepStatus.COMPLETED)
        assert step.is_complete()
        assert step.is_successful()
        assert not step.is_failed()
        assert not step.is_skipped()
        assert not step.is_pending()
        assert not step.is_in_progress()
        
        # Update status to FAILED
        step.update_status(PlanStepStatus.FAILED)
        assert step.is_complete()
        assert not step.is_successful()
        assert step.is_failed()
        assert not step.is_skipped()
        assert not step.is_pending()
        assert not step.is_in_progress()
        
        # Update status to SKIPPED
        step.update_status(PlanStepStatus.SKIPPED)
        assert step.is_complete()
        assert not step.is_successful()
        assert not step.is_failed()
        assert step.is_skipped()
        assert not step.is_pending()
        assert not step.is_in_progress()
    
    def test_dependency_methods(self):
        """Test dependency methods."""
        step = PlanStep(
            task_description="Test task",
        )
        
        # Initially no dependencies
        assert not step.has_dependency("step1")
        
        # Add a dependency
        step.add_dependency("step1")
        assert step.has_dependency("step1")
        assert step.dependencies == ["step1"]
        
        # Add another dependency
        step.add_dependency("step2")
        assert step.has_dependency("step2")
        assert set(step.dependencies) == {"step1", "step2"}
        
        # Add a duplicate dependency
        step.add_dependency("step1")
        assert set(step.dependencies) == {"step1", "step2"}
        
        # Remove a dependency
        step.remove_dependency("step1")
        assert not step.has_dependency("step1")
        assert step.dependencies == ["step2"]
        
        # Remove a non-existent dependency
        step.remove_dependency("step3")
        assert step.dependencies == ["step2"]
    
    def test_complete_method(self):
        """Test complete method."""
        step = PlanStep(
            task_description="Test task",
        )
        
        # Complete the step
        step.complete(
            result="Test result",
            actual_cost=0.2,
            actual_tokens=2000,
        )
        
        assert step.status == PlanStepStatus.COMPLETED
        assert step.result == "Test result"
        assert step.actual_cost == 0.2
        assert step.actual_tokens == 2000
    
    def test_fail_method(self):
        """Test fail method."""
        step = PlanStep(
            task_description="Test task",
        )
        
        # Fail the step
        step.fail(
            error="Test error",
            actual_cost=0.2,
            actual_tokens=2000,
        )
        
        assert step.status == PlanStepStatus.FAILED
        assert step.error == "Test error"
        assert step.actual_cost == 0.2
        assert step.actual_tokens == 2000
        
        # Fail the step without cost and tokens
        step = PlanStep(
            task_description="Test task",
        )
        step.fail(error="Test error")
        
        assert step.status == PlanStepStatus.FAILED
        assert step.error == "Test error"
        assert step.actual_cost is None
        assert step.actual_tokens is None
    
    def test_skip_method(self):
        """Test skip method."""
        step = PlanStep(
            task_description="Test task",
        )
        
        # Skip the step
        step.skip(reason="Test reason")
        
        assert step.status == PlanStepStatus.SKIPPED
        assert step.error == "Test reason"
        assert step.actual_cost == 0.0
        assert step.actual_tokens == 0
    
    def test_start_method(self):
        """Test start method."""
        step = PlanStep(
            task_description="Test task",
        )
        
        # Start the step
        step.start()
        
        assert step.status == PlanStepStatus.IN_PROGRESS
    
    def test_reset_method(self):
        """Test reset method."""
        step = PlanStep(
            task_description="Test task",
        )
        
        # Complete the step
        step.complete(
            result="Test result",
            actual_cost=0.2,
            actual_tokens=2000,
        )
        
        # Reset the step
        step.reset()
        
        assert step.status == PlanStepStatus.PENDING
        assert step.result is None
        assert step.error is None
        assert step.actual_cost is None
        assert step.actual_tokens is None
    
    def test_get_cost_difference(self):
        """Test get_cost_difference method."""
        step = PlanStep(
            task_description="Test task",
            estimated_cost=0.1,
        )
        
        # No actual cost yet
        assert step.get_cost_difference() is None
        
        # Set actual cost
        step.actual_cost = 0.2
        assert step.get_cost_difference() == 0.1
        
        # Set actual cost lower than estimated
        step.actual_cost = 0.05
        assert step.get_cost_difference() == -0.05
    
    def test_get_token_difference(self):
        """Test get_token_difference method."""
        step = PlanStep(
            task_description="Test task",
            estimated_tokens=1000,
        )
        
        # No actual tokens yet
        assert step.get_token_difference() is None
        
        # Set actual tokens
        step.actual_tokens = 2000
        assert step.get_token_difference() == 1000
        
        # Set actual tokens lower than estimated
        step.actual_tokens = 500
        assert step.get_token_difference() == -500
    
    def test_to_dict_and_from_dict(self):
        """Test to_dict and from_dict methods."""
        step = PlanStep(
            task_description="Test task",
            step_type=StepType.TASK,
            priority=StepPriority.MEDIUM,
            estimated_cost=0.1,
            estimated_tokens=1000,
            dependencies=["step1", "step2"],
            metadata={"key": "value"},
        )
        
        # Convert to dict
        data = step.to_dict()
        
        # Check dict contents
        assert data["task_description"] == "Test task"
        assert data["step_type"] == "task"
        assert data["priority"] == "medium"
        assert data["estimated_cost"] == 0.1
        assert data["estimated_tokens"] == 1000
        assert data["dependencies"] == ["step1", "step2"]
        assert data["metadata"] == {"key": "value"}
        
        # Convert back to PlanStep
        new_step = PlanStep.from_dict(data)
        
        # Check new step
        assert new_step.task_description == "Test task"
        assert new_step.step_type == StepType.TASK
        assert new_step.priority == StepPriority.MEDIUM
        assert new_step.estimated_cost == 0.1
        assert new_step.estimated_tokens == 1000
        assert new_step.dependencies == ["step1", "step2"]
        assert new_step.metadata == {"key": "value"}
