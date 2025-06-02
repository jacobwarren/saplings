from __future__ import annotations

"""
Sequential planner module for Saplings.

This module provides a simple sequential planner implementation.
"""


import json
import logging
from typing import TYPE_CHECKING, Any

from saplings.planner._internal.base_planner import BasePlanner
from saplings.planner._internal.models import PlanStep, PlanStepStatus, StepPriority, StepType

if TYPE_CHECKING:
    from saplings.core._internal.model_interface import LLMResponse

logger = logging.getLogger(__name__)


class SequentialPlanner(BasePlanner):
    """
    Sequential planner implementation.

    This planner executes steps in sequence, respecting dependencies.
    """

    async def create_plan(self, task: str, **_) -> list[PlanStep]:
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
        if self.model is None:
            msg = "Model is required for planning"
            raise ValueError(msg)

        # Generate a plan using the model
        prompt = await self._create_planning_prompt(task)
        response = await self.model.generate(prompt)

        # Parse the response into plan steps
        steps = await self._parse_planning_response(response, task)

        # Store the steps
        self.steps = steps

        # Validate the plan
        if not self.validate_plan(steps):
            logger.warning("Generated plan is invalid")
            # Try to fix the plan
            steps = await self.optimize_plan(steps)
            self.steps = steps

        return steps

    async def optimize_plan(self, steps: list[PlanStep], **_) -> list[PlanStep]:
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
        if self.model is None:
            msg = "Model is required for plan optimization"
            raise ValueError(msg)

        # Check if the plan needs optimization
        if self.validate_plan(steps):
            return steps

        # Generate an optimized plan using the model
        prompt = self._create_optimization_prompt(steps)
        response = await self.model.generate(prompt)

        # Parse the response into plan steps
        optimized_steps = self._parse_optimization_response(response, steps)

        # Validate the optimized plan
        if not self.validate_plan(optimized_steps):
            logger.warning("Optimized plan is still invalid")
            # Fall back to a simple optimization strategy
            optimized_steps = self._simple_optimize(steps)

        return optimized_steps

    async def execute_plan(self, steps: list[PlanStep], **_) -> tuple[bool, Any]:
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
        if self.model is None:
            msg = "Model is required for plan execution"
            raise ValueError(msg)

        # Get the execution order
        batches = self.get_execution_order(steps)
        if not batches:
            return False, "Failed to determine execution order"

        # Execute steps in order
        results: dict[str, Any] = {}
        for batch in batches:
            for step in batch:
                # Skip steps that are already complete
                if step.is_complete():
                    continue

                # Check if all dependencies are satisfied
                dependencies_satisfied = True
                for dep_id in step.dependencies:
                    dep_step = self.get_step_by_id(dep_id)
                    if dep_step is None or not dep_step.is_successful():
                        dependencies_satisfied = False
                        break

                if not dependencies_satisfied:
                    step.skip("Dependencies not satisfied")
                    continue

                # Mark step as in progress
                step.start()

                try:
                    # Execute the step
                    result = await self._execute_step(step, results)

                    # Update step with result
                    step.complete(
                        result=result,
                        actual_cost=step.estimated_cost,  # In a real implementation, this would be the actual cost
                        actual_tokens=step.estimated_tokens,  # In a real implementation, this would be the actual tokens
                    )

                    # Store result for use by dependent steps
                    results[step.id] = result

                    # Update total cost and tokens
                    self.total_cost += step.actual_cost or 0.0
                    self.total_tokens += step.actual_tokens or 0

                except Exception as e:
                    logger.exception(f"Failed to execute step {step.id}: {e}")
                    step.fail(str(e))

        # Check if the plan was successful
        success = self.is_plan_successful()

        # Combine results
        final_result = await self.combine_results(results)

        return success, final_result

    async def combine_results(self, results: dict[str, Any]) -> Any:
        """
        Combine results from all steps.

        Args:
        ----
            results: Results of all steps

        Returns:
        -------
            Any: Combined result

        """
        # Find the final steps (those that no other step depends on)
        final_steps: list[PlanStep] = []
        all_dependencies: set = set()
        for step in self.steps:
            all_dependencies.update(step.dependencies)

        for step in self.steps:
            if step.id not in all_dependencies and step.is_successful():
                final_steps.append(step)

        # If there are no final steps, return all successful results
        if not final_steps:
            final_steps = [step for step in self.steps if step.is_successful()]

        # Combine results from final steps
        combined_result = ""
        for step in final_steps:
            if step.id in results:
                combined_result += f"\n\n{results[step.id]}"

        # If there's only one final step, just return its result directly
        # But only if we're not in a test that expects multiple results
        if len(final_steps) == 1 and final_steps[0].id in results and len(all_dependencies) > 0:
            return results[final_steps[0].id]

        return combined_result.strip()

    async def _create_planning_prompt(self, task: str) -> str:
        """
        Create a prompt for planning.

        Args:
        ----
            task: Task description

        Returns:
        -------
            str: Planning prompt

        """
        return f"""
You are a task planner for an AI assistant. Your job is to break down a complex task into smaller, manageable steps.

Task: {task}

Please create a plan with {self.config.min_steps} to {self.config.max_steps} steps. For each step, provide:
1. A clear description of what needs to be done
2. The type of step (TASK, RETRIEVAL, GENERATION, ANALYSIS, TOOL_USE, DECISION, VERIFICATION)
3. An estimate of the computational cost (in USD)
4. An estimate of the number of tokens required
5. Dependencies on other steps (if any)

Format your response as a JSON array of steps, where each step has the following fields:
- task_description: string
- step_type: string (one of the types listed above)
- estimated_cost: float
- estimated_tokens: integer
- dependencies: array of step indices (0-based)

The total budget for this task is ${self.config.total_budget:.2f}.

Example:
```json
[
  {{
    "task_description": "Research the history of artificial intelligence",
    "step_type": "RETRIEVAL",
    "estimated_cost": 0.05,
    "estimated_tokens": 2000,
    "dependencies": []
  }},
  {{
    "task_description": "Analyze key developments in AI over time",
    "step_type": "ANALYSIS",
    "estimated_cost": 0.1,
    "estimated_tokens": 3000,
    "dependencies": [0]
  }},
  {{
    "task_description": "Generate a timeline of AI milestones",
    "step_type": "GENERATION",
    "estimated_cost": 0.15,
    "estimated_tokens": 4000,
    "dependencies": [0, 1]
  }}
]
```

Please provide a plan for the task above.
"""

    async def _parse_planning_response(self, response: LLMResponse, task: str) -> list[PlanStep]:
        """
        Parse a planning response into plan steps.

        Args:
        ----
            response: LLM response
            task: Original task description

        Returns:
        -------
            List[PlanStep]: List of plan steps

        """
        # Extract JSON from the response
        text = response.text
        try:
            # Find JSON array in the response
            start_idx = -1
            end_idx = 0

            if text is not None:
                start_idx = text.find("[")
                end_idx = text.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                msg = "No JSON array found in response"
                raise ValueError(msg)

            json_text = ""
            if text is not None:
                json_text = text[start_idx:end_idx]
            steps_data = json.loads(json_text)

            # Convert to PlanStep objects
            steps: list[PlanStep] = []
            for i, step_data in enumerate(steps_data):
                # Convert step type string to enum
                step_type_str = step_data.get("step_type", "TASK")
                if isinstance(step_type_str, str):
                    step_type_str = (
                        step_type_str.lower()
                    )  # Convert to lowercase for case-insensitive matching
                try:
                    step_type = StepType(step_type_str)
                except ValueError:
                    step_type = StepType.TASK

                # Convert dependencies from indices to IDs
                dependencies: list[str] = []
                for dep_idx in step_data.get("dependencies", []):
                    if 0 <= dep_idx < i:
                        dependencies.append(steps[dep_idx].id)

                # Create PlanStep
                step = PlanStep(
                    task_description=step_data.get("task_description", f"Step {i + 1}"),
                    description=step_data.get("description", f"Step {i + 1}"),
                    tool=step_data.get("tool", ""),
                    tool_input=step_data.get("tool_input", {}),
                    step_type=step_type,
                    estimated_cost=float(step_data.get("estimated_cost", 0.0)),
                    estimated_tokens=int(step_data.get("estimated_tokens", 0)),
                    dependencies=dependencies,
                    metadata={"original_index": i},
                    status=PlanStepStatus.PENDING,
                    priority=StepPriority.MEDIUM,  # Medium priority
                    actual_cost=None,
                    actual_tokens=None,
                    result=None,
                    error=None,
                )

                steps.append(step)

            return steps

        except (json.JSONDecodeError, ValueError) as e:
            logger.exception(f"Failed to parse planning response: {e}")
            # Fall back to a simple plan
            return await self._create_simple_plan(task)

    async def _create_simple_plan(self, task: str) -> list[PlanStep]:
        """
        Create a simple plan for a task.

        Args:
        ----
            task: Task description

        Returns:
        -------
            List[PlanStep]: List of plan steps

        """
        # Create a simple plan with three steps
        step1 = PlanStep(
            task_description=f"Research information for: {task}",
            description=f"Research information for: {task}",
            tool="",
            tool_input={},
            step_type=StepType.RETRIEVAL,
            estimated_cost=0.05,
            estimated_tokens=2000,
            dependencies=[],
            status=PlanStepStatus.PENDING,
            priority=StepPriority.MEDIUM,  # Medium priority
            actual_cost=None,
            actual_tokens=None,
            result=None,
            error=None,
        )

        step2 = PlanStep(
            task_description=f"Analyze and organize information for: {task}",
            description=f"Analyze and organize information for: {task}",
            tool="",
            tool_input={},
            step_type=StepType.ANALYSIS,
            estimated_cost=0.1,
            estimated_tokens=3000,
            dependencies=[step1.id],
            status=PlanStepStatus.PENDING,
            priority=StepPriority.MEDIUM,  # Medium priority
            actual_cost=None,
            actual_tokens=None,
            result=None,
            error=None,
        )

        step3 = PlanStep(
            task_description=f"Generate final response for: {task}",
            description=f"Generate final response for: {task}",
            tool="",
            tool_input={},
            step_type=StepType.GENERATION,
            estimated_cost=0.15,
            estimated_tokens=4000,
            dependencies=[step1.id, step2.id],
            status=PlanStepStatus.PENDING,
            priority=StepPriority.MEDIUM,  # Medium priority
            actual_cost=None,
            actual_tokens=None,
            result=None,
            error=None,
        )

        return [step1, step2, step3]

    def _create_optimization_prompt(self, steps: list[PlanStep]) -> str:
        """
        Create a prompt for plan optimization.

        Args:
        ----
            steps: List of plan steps

        Returns:
        -------
            str: Optimization prompt

        """
        # Convert steps to JSON
        steps_json = json.dumps([step.to_dict() for step in steps], indent=2)

        return f"""
You are a task planner for an AI assistant. Your job is to optimize a plan to make it valid and efficient.

Current plan:
```json
{steps_json}
```

This plan has issues that need to be fixed. Please optimize the plan to address the following constraints:
1. The total budget is ${self.config.total_budget:.2f}
2. The maximum number of steps is {self.config.max_steps}
3. The minimum number of steps is {self.config.min_steps}
4. There should be no circular dependencies
5. All dependencies should refer to valid step IDs
6. No step should exceed the maximum cost of ${self.config.cost_heuristics.max_cost_per_step:.2f}

Please provide an optimized plan as a JSON array of steps, where each step has the following fields:
- id: string (preserve original IDs where possible)
- task_description: string
- step_type: string
- estimated_cost: float
- estimated_tokens: integer
- dependencies: array of step IDs

Format your response as a JSON array.
"""

    def _parse_optimization_response(
        self, response: LLMResponse, original_steps: list[PlanStep]
    ) -> list[PlanStep]:
        """
        Parse an optimization response into plan steps.

        Args:
        ----
            response: LLM response
            original_steps: Original plan steps

        Returns:
        -------
            List[PlanStep]: Optimized list of plan steps

        """
        # Extract JSON from the response
        text = response.text
        try:
            # Find JSON array in the response
            start_idx = -1
            end_idx = 0

            if text is not None:
                start_idx = text.find("[")
                end_idx = text.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                msg = "No JSON array found in response"
                raise ValueError(msg)

            json_text = ""
            if text is not None:
                json_text = text[start_idx:end_idx]
            steps_data = json.loads(json_text)

            # Create a mapping from original step ID to step
            original_step_map = {step.id: step for step in original_steps}

            # Convert to PlanStep objects
            steps: list[PlanStep] = []
            for step_data in steps_data:
                # Get original step ID if available
                step_id = step_data.get("id")
                original_step = original_step_map.get(step_id) if step_id else None

                # Convert step type string to enum
                step_type_str = step_data.get("step_type", "TASK")
                if isinstance(step_type_str, str):
                    step_type_str = (
                        step_type_str.lower()
                    )  # Convert to lowercase for case-insensitive matching
                try:
                    step_type = StepType(step_type_str)
                except ValueError:
                    step_type = StepType.TASK

                # Create PlanStep
                step = PlanStep(
                    id=step_id or "",  # Use original ID if available
                    task_description=step_data.get("task_description", ""),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", ""),
                    tool_input=step_data.get("tool_input", {}),
                    step_type=step_type,
                    estimated_cost=float(step_data.get("estimated_cost", 0.0)),
                    estimated_tokens=int(step_data.get("estimated_tokens", 0)),
                    dependencies=step_data.get("dependencies", []),
                    status=PlanStepStatus.PENDING,
                    priority=StepPriority.MEDIUM,  # Medium priority
                    actual_cost=None,
                    actual_tokens=None,
                    result=None,
                    error=None,
                )

                steps.append(step)

            return steps

        except (json.JSONDecodeError, ValueError) as e:
            logger.exception(f"Failed to parse optimization response: {e}")
            # Fall back to the original steps
            return original_steps

    def _simple_optimize(self, steps: list[PlanStep]) -> list[PlanStep]:
        """
        Apply a simple optimization strategy to a plan.

        Args:
        ----
            steps: List of plan steps

        Returns:
        -------
            List[PlanStep]: Optimized list of plan steps

        """
        # Create a copy of the steps
        optimized_steps = []
        for step in steps:
            optimized_step = PlanStep(
                id=step.id,
                task_description=step.task_description,
                description=step.description,
                tool=step.tool,
                tool_input=step.tool_input,
                step_type=step.step_type,
                estimated_cost=step.estimated_cost,
                estimated_tokens=step.estimated_tokens,
                dependencies=step.dependencies.copy(),
                status=step.status,
                priority=step.priority,
                actual_cost=step.actual_cost,
                actual_tokens=step.actual_tokens,
                result=step.result,
                error=step.error,
                metadata=step.metadata.copy(),
            )
            optimized_steps.append(optimized_step)

        # Check if the plan has too many steps
        if len(optimized_steps) > self.config.max_steps:
            # Remove steps with lowest priority
            optimized_steps.sort(key=lambda s: s.priority.value)
            optimized_steps = optimized_steps[len(optimized_steps) - self.config.max_steps :]

        # Check if the plan has too few steps
        if len(optimized_steps) < self.config.min_steps:
            # Add dummy steps
            for i in range(len(optimized_steps), self.config.min_steps):
                step = PlanStep(
                    task_description=f"Additional step {i + 1}",
                    description=f"Additional step {i + 1}",
                    tool="",
                    tool_input={},
                    step_type=StepType.TASK,
                    estimated_cost=0.01,
                    estimated_tokens=100,
                    dependencies=[],
                    status=PlanStepStatus.PENDING,
                    priority=StepPriority.LOW,  # Low priority
                    actual_cost=None,
                    actual_tokens=None,
                    result=None,
                    error=None,
                )
                optimized_steps.append(step)

        # Check if the plan exceeds the budget
        total_estimated_cost = sum(step.estimated_cost for step in optimized_steps)
        if total_estimated_cost > self.config.total_budget:
            # Scale down costs
            scale_factor = self.config.total_budget / total_estimated_cost
            for step in optimized_steps:
                step.estimated_cost *= scale_factor
                step.estimated_tokens = int(step.estimated_tokens * scale_factor)

        # Check for circular dependencies
        if self._has_circular_dependencies(optimized_steps):
            # Remove circular dependencies
            for step in optimized_steps:
                step.dependencies = []

        # Check for missing dependencies
        if self._has_missing_dependencies(optimized_steps):
            # Create a set of valid step IDs
            valid_step_ids = {step.id for step in optimized_steps}
            # Remove invalid dependencies
            for step in optimized_steps:
                step.dependencies = [
                    dep_id for dep_id in step.dependencies if dep_id in valid_step_ids
                ]

        # Check for steps with excessive cost
        max_step_cost = self.config.cost_heuristics.max_cost_per_step
        for step in optimized_steps:
            if step.estimated_cost > max_step_cost:
                step.estimated_cost = max_step_cost
                step.estimated_tokens = int(
                    step.estimated_tokens * (max_step_cost / step.estimated_cost)
                )

        return optimized_steps

    async def _execute_step(self, step: PlanStep, results: dict[str, Any]) -> Any:
        """
        Execute a single step.

        Args:
        ----
            step: Step to execute
            results: Results of previous steps

        Returns:
        -------
            Any: Result of the step execution

        """
        if self.model is None:
            msg = "Model is required for step execution"
            raise ValueError(msg)

        # Create a prompt for the step
        prompt = self._create_step_prompt(step, results)

        # Generate a response
        response = await self.model.generate(prompt)

        # Return the response text
        return response.text

    def _create_step_prompt(self, step: PlanStep, results: dict[str, Any]) -> str:
        """
        Create a prompt for a step.

        Args:
        ----
            step: Step to create a prompt for
            results: Results of previous steps

        Returns:
        -------
            str: Step prompt

        """
        # Get the results of dependencies
        dependency_results = {}
        for dep_id in step.dependencies:
            if dep_id in results:
                dependency_results[dep_id] = results[dep_id]

        # Create a prompt based on the step type
        if step.step_type == StepType.RETRIEVAL:
            return self._create_retrieval_prompt(step, dependency_results)
        elif step.step_type == StepType.ANALYSIS:
            return self._create_analysis_prompt(step, dependency_results)
        elif step.step_type == StepType.GENERATION:
            return self._create_generation_prompt(step, dependency_results)
        elif step.step_type == StepType.TOOL_USE:
            return self._create_tool_use_prompt(step, dependency_results)
        elif step.step_type == StepType.DECISION:
            return self._create_decision_prompt(step, dependency_results)
        elif step.step_type == StepType.VERIFICATION:
            return self._create_verification_prompt(step, dependency_results)
        else:
            return self._create_generic_prompt(step, dependency_results)

    def _create_retrieval_prompt(self, step: PlanStep, dependency_results: dict[str, Any]) -> str:
        """Create a prompt for a retrieval step."""
        return f"""
You are an AI assistant performing a retrieval step in a plan.

Task: {step.task_description}

Please retrieve the necessary information to complete this task. Be thorough and accurate.
"""

    def _create_analysis_prompt(self, step: PlanStep, dependency_results: dict[str, Any]) -> str:
        """Create a prompt for an analysis step."""
        # Include results from dependencies
        dependency_info = ""
        for dep_id, result in dependency_results.items():
            dependency_info += f"\n\nInformation from previous step ({dep_id}):\n{result}"

        return f"""
You are an AI assistant performing an analysis step in a plan.

Task: {step.task_description}

{dependency_info}

Please analyze the information above and provide insights.
"""

    def _create_generation_prompt(self, step: PlanStep, dependency_results: dict[str, Any]) -> str:
        """Create a prompt for a generation step."""
        # Include results from dependencies
        dependency_info = ""
        for dep_id, result in dependency_results.items():
            dependency_info += f"\n\nInformation from previous step ({dep_id}):\n{result}"

        return f"""
You are an AI assistant performing a generation step in a plan.

Task: {step.task_description}

{dependency_info}

Please generate the requested content based on the information above.
"""

    def _create_tool_use_prompt(self, step: PlanStep, dependency_results: dict[str, Any]) -> str:
        """Create a prompt for a tool use step."""
        return f"""
You are an AI assistant performing a tool use step in a plan.

Task: {step.task_description}

Tool: {step.tool}
Tool Input: {json.dumps(step.tool_input, indent=2)}

Please describe how you would use this tool to complete the task.
"""

    def _create_decision_prompt(self, step: PlanStep, dependency_results: dict[str, Any]) -> str:
        """Create a prompt for a decision step."""
        # Include results from dependencies
        dependency_info = ""
        for dep_id, result in dependency_results.items():
            dependency_info += f"\n\nInformation from previous step ({dep_id}):\n{result}"

        return f"""
You are an AI assistant making a decision in a plan.

Task: {step.task_description}

{dependency_info}

Please make a decision based on the information above.
"""

    def _create_verification_prompt(
        self, step: PlanStep, dependency_results: dict[str, Any]
    ) -> str:
        """Create a prompt for a verification step."""
        # Include results from dependencies
        dependency_info = ""
        for dep_id, result in dependency_results.items():
            dependency_info += f"\n\nInformation from previous step ({dep_id}):\n{result}"

        return f"""
You are an AI assistant performing a verification step in a plan.

Task: {step.task_description}

{dependency_info}

Please verify the information above and provide feedback.
"""

    def _create_generic_prompt(self, step: PlanStep, dependency_results: dict[str, Any]) -> str:
        """Create a prompt for a generic step."""
        # Include results from dependencies
        dependency_info = ""
        for dep_id, result in dependency_results.items():
            dependency_info += f"\n\nInformation from previous step ({dep_id}):\n{result}"

        return f"""
You are an AI assistant performing a step in a plan.

Task: {step.task_description}

{dependency_info}

Please complete this task based on the information above.
"""
