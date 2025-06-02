from __future__ import annotations

"""
Judge service for Saplings.

This module provides a service for judging output quality using rubrics.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from saplings.api.core.interfaces import IJudgeService
from saplings.core._internal.model_adapter import LLM
from saplings.core.events import CoreEvent, CoreEventType, get_event_bus
from saplings.core.resilience import DEFAULT_TIMEOUT, with_timeout

logger = logging.getLogger(__name__)


class JudgeService(IJudgeService):
    """Service that manages judging of outputs."""

    def __init__(
        self,
        model: "LLM",
        trace_manager: Any = None,
    ) -> None:
        """
        Initialize the judge service.

        Args:
        ----
            model: The model to use for judging
            trace_manager: Optional trace manager for monitoring

        Note:
        ----
            This constructor initializes the service but does not create the JudgeAgent.
            The JudgeAgent will be created lazily when needed, or you can use the
            JudgeServiceBuilder.build_async() method to initialize it eagerly.

        """
        self._model = model
        self._trace_manager = trace_manager
        self._judge: Any = None  # Will be initialized to JudgeAgent
        self._judgment_history: List[Dict[str, Any]] = []
        self._event_bus = get_event_bus()
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

        # Register for events
        self._register_event_handlers()

        logger.info("JudgeService initialized")

    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # No event handlers needed yet, but this method can be used
        # to register handlers for events from other services

    async def initialize_judge(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> None:
        """
        Initialize the judge.

        This method is thread-safe and will only initialize the judge once.
        If called multiple times, subsequent calls will return immediately.

        Args:
        ----
            timeout: Optional timeout in seconds

        Raises:
        ------
            TimeoutError: If initialization times out

        """
        # Check if already initialized
        if self._initialized and self._judge is not None:
            logger.debug("Judge already initialized")
            return

        # Use a lock to prevent multiple initializations
        async with self._initialization_lock:
            # Check again in case another task initialized while we were waiting
            if self._initialized and self._judge is not None:
                logger.debug("Judge already initialized by another task")
                return

            # Define initialization function
            async def _init_judge() -> None:
                try:
                    # Import from the public API to avoid circular imports
                    from saplings.api.judge import JudgeAgent

                    # Create a judge agent with lazy loading of rubrics
                    self._judge = JudgeAgent(
                        model=self._model,
                        rubric_path=None,  # Use default rubric
                        lazy_load_rubrics=True,  # Enable lazy loading of rubrics
                    )

                    # Mark as initialized
                    self._initialized = True

                    # Publish event
                    self._event_bus.publish(
                        CoreEvent(
                            event_type=CoreEventType.JUDGE_INITIALIZED,
                            data={"judge_id": id(self._judge)},
                            source="JudgeService",
                        )
                    )

                    logger.info("Judge initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize judge: {e}")
                    raise

            # Execute with timeout
            await with_timeout(_init_judge(), timeout=timeout, operation_name="initialize_judge")

    async def judge(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        judgment_type: str = "general",
        trace_id: Optional[str] = None,
    ) -> JudgmentResult:
        """
        Judge output quality.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to judge
            judgment_type: Type of judgment to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            JudgmentResult: Judgment results

        Raises:
        ------
            ValueError: If the judge is not initialized

        """
        # Start span if tracing is enabled
        span = None
        if self._trace_manager and trace_id:
            span = self._trace_manager.start_span(
                name="judge_output",
                trace_id=trace_id,
                attributes={
                    "component": "judge",
                    "judgment_type": judgment_type,
                },
            )

        # Publish event
        self._event_bus.publish(
            CoreEvent(
                event_type=CoreEventType.JUDGE_REQUESTED,
                data={
                    "judgment_type": judgment_type,
                    "input_length": len(str(input_data)),
                    "output_length": len(str(output_data)),
                },
                source="JudgeService",
                trace_id=trace_id,
            )
        )

        try:
            # Ensure judge is initialized
            await self.initialize_judge()

            # Extract prompt from input_data if available
            prompt = input_data.get("prompt", "")
            if not prompt and "task" in input_data:
                prompt = input_data.get("task", "")

            # Convert output_data to string if it's not already
            output = str(output_data) if not isinstance(output_data, str) else output_data

            # Judge the output (we know self._judge is not None because initialize_judge was called)
            judgment = await self._judge.judge(
                output=output,
                prompt=prompt,
            )

            # Convert to JudgmentResult
            result = JudgmentResult(
                score=judgment.overall_score,
                feedback=judgment.critique,
                strengths=judgment.strengths or [],
                weaknesses=judgment.weaknesses or [],
                details={
                    "dimension_scores": judgment.dimension_scores,
                    "metadata": judgment.metadata,
                },
            )

            # Add to history
            self._judgment_history.append(
                {
                    "input": input_data,
                    "output": output_data,
                    "judgment_type": judgment_type,
                    "result": result,
                }
            )

            # Publish event
            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.JUDGE_COMPLETED,
                    data={
                        "judgment_type": judgment_type,
                        "score": result.score,
                    },
                    source="JudgeService",
                    trace_id=trace_id,
                )
            )

            return result
        except Exception as e:
            # Publish error event
            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.JUDGE_FAILED,
                    data={"error": str(e)},
                    source="JudgeService",
                    trace_id=trace_id,
                )
            )
            raise
        finally:
            # End span if tracing is enabled
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    async def create_rubric(
        self, name: str, description: str, criteria: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a judgment rubric.

        Args:
        ----
            name: Rubric name
            description: Rubric description
            criteria: List of criteria definitions

        Returns:
        -------
            Dict[str, Any]: The created rubric

        """
        # Ensure judge is initialized
        await self.initialize_judge()

        # Import from the public API to avoid circular imports
        from saplings.api.judge import Rubric, RubricItem, ScoringDimension

        # Create the rubric with the provided criteria
        rubric_items = []
        for criterion in criteria:
            dimension_str = criterion.get("dimension", "RELEVANCE")
            try:
                dimension = ScoringDimension[dimension_str.upper()]
            except (KeyError, AttributeError):
                dimension = ScoringDimension.RELEVANCE

            rubric_items.append(
                RubricItem(
                    dimension=dimension,
                    weight=criterion.get("weight", 1.0),
                    description=criterion.get("description", ""),
                    criteria=criterion.get("criteria", {}),
                )
            )

        rubric = Rubric(
            name=name,
            description=description,
            items=rubric_items,
        )

        # Return the rubric as a dictionary
        return {
            "name": rubric.name,
            "description": rubric.description,
            "criteria": criteria,  # Use the input criteria directly
        }

    async def judge_with_rubric(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        rubric_name: str,
        trace_id: Optional[str] = None,
    ) -> JudgmentResult:
        """
        Judge output using a specific rubric.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to judge
            rubric_name: Name of the rubric to use
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            JudgmentResult: Judgment results

        """
        # Start span if tracing is enabled
        span = None
        if self._trace_manager and trace_id:
            span = self._trace_manager.start_span(
                name="judge_with_rubric",
                trace_id=trace_id,
                attributes={
                    "component": "judge",
                    "rubric_name": rubric_name,
                },
            )

        try:
            # Ensure judge is initialized
            await self.initialize_judge()

            # Extract prompt from input_data if available
            prompt = input_data.get("prompt", "")
            if not prompt and "task" in input_data:
                prompt = input_data.get("task", "")

            # Convert output_data to string if it's not already
            output = str(output_data) if not isinstance(output_data, str) else output_data

            # Import the Rubric class from the public API
            from saplings.api.judge import Rubric

            # Create a rubric based on the template name
            # In a real implementation, this would load the rubric from a registry or database
            rubric = Rubric.default()
            rubric.name = rubric_name

            # Judge the output with the specified rubric
            judgment = await self._judge.judge(
                output=output,
                prompt=prompt,
                rubric=rubric,
            )

            # Convert to JudgmentResult
            result = JudgmentResult(
                score=judgment.overall_score,
                feedback=judgment.critique,
                strengths=judgment.strengths or [],
                weaknesses=judgment.weaknesses or [],
                details={
                    "dimension_scores": judgment.dimension_scores,
                    "metadata": judgment.metadata,
                },
            )

            return result
        finally:
            # End span if tracing is enabled
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    def get_judgment_history(self) -> List[Dict[str, Any]]:
        """
        Get judgment history.

        Returns
        -------
            List[Dict[str, Any]]: History of judgment operations

        """
        return self._judgment_history
