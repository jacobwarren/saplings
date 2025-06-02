from __future__ import annotations

"""
Judge strategy module for Saplings.

This module provides strategy interfaces and implementations for judging outputs.
It follows the Strategy pattern to allow different judging strategies to be used
without changing the client code.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TypeVar

from saplings.api.core.interfaces import IJudgeService


@dataclass
class JudgmentResult:
    """Standard result for judgment operations."""

    score: float
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    details: Optional[Dict[str, Any]] = None


logger = logging.getLogger(__name__)

T = TypeVar("T")


class IJudgeStrategy(Protocol):
    """Interface for judge strategies."""

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

        """
        ...

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
        ...


class DirectJudgeStrategy(IJudgeStrategy):
    """
    Strategy that uses a JudgeAgent directly.

    This strategy initializes and uses a JudgeAgent directly without
    going through the JudgeService.
    """

    def __init__(self, model: Any, rubric_path: Optional[str] = None):
        """
        Initialize the direct judge strategy.

        Args:
        ----
            model: The model to use for judging
            rubric_path: Optional path to a rubric file

        """
        self._model = model
        self._rubric_path = rubric_path
        self._judge = None
        logger.info("DirectJudgeStrategy initialized")

    async def _ensure_judge_initialized(self) -> None:
        """
        Ensure the judge is initialized.

        Raises
        ------
            ValueError: If judge initialization fails

        """
        if self._judge is None:
            try:
                # Import here to avoid circular imports
                from saplings.judge._internal.judge_agent import JudgeAgent

                # Create a judge agent
                self._judge = JudgeAgent(
                    model=self._model,
                    rubric_path=self._rubric_path,
                )
                logger.info("Judge initialized")
            except Exception as e:
                logger.error(f"Failed to initialize judge: {e}")
                raise ValueError(f"Failed to initialize judge: {e}")

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

        """
        await self._ensure_judge_initialized()

        # Extract prompt from input_data if available
        prompt = input_data.get("prompt", "")
        if not prompt and "task" in input_data:
            prompt = input_data.get("task", "")

        # Convert output_data to string if needed
        output_text = str(output_data)
        if hasattr(output_data, "text"):
            output_text = output_data.text

        # Judge the output
        # The JudgeAgent.judge method only accepts prompt and output parameters
        judge_result = await self._judge.judge(
            prompt=prompt,
            output=output_text,
        )

        # Convert JudgeResult to JudgmentResult
        result = JudgmentResult(
            score=judge_result.overall_score,
            feedback=judge_result.critique,
            strengths=[
                f"{score.dimension.value}: {score.explanation}"
                for score in judge_result.dimension_scores
                if score.score >= 0.7
            ],
            weaknesses=[
                f"{score.dimension.value}: {score.explanation}"
                for score in judge_result.dimension_scores
                if score.score < 0.7
            ],
            details={
                "passed": judge_result.passed,
                "trace_id": trace_id,
                "judgment_type": judgment_type,
                **(judge_result.metadata or {}),
            },
        )

        return result

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
        await self._ensure_judge_initialized()

        # Extract prompt from input_data if available
        prompt = input_data.get("prompt", "")
        if not prompt and "task" in input_data:
            prompt = input_data.get("task", "")

        # Convert output_data to string if needed
        output_text = str(output_data)
        if hasattr(output_data, "text"):
            output_text = output_data.text

        # Judge the output with the specified rubric
        # The JudgeAgent doesn't have a direct method for judging with a named rubric
        # We'll use the regular judge method and add the rubric name to metadata
        judge_result = await self._judge.judge(
            prompt=prompt,
            output=output_text,
        )

        # Convert JudgeResult to JudgmentResult
        result = JudgmentResult(
            score=judge_result.overall_score,
            feedback=judge_result.critique,
            strengths=[
                f"{score.dimension.value}: {score.explanation}"
                for score in judge_result.dimension_scores
                if score.score >= 0.7
            ],
            weaknesses=[
                f"{score.dimension.value}: {score.explanation}"
                for score in judge_result.dimension_scores
                if score.score < 0.7
            ],
            details={
                "passed": judge_result.passed,
                "trace_id": trace_id,
                "rubric_name": rubric_name,
                **(judge_result.metadata or {}),
            },
        )

        return result


class ServiceJudgeStrategy(IJudgeStrategy):
    """
    Strategy that uses the JudgeService.

    This strategy delegates to the JudgeService for judging outputs.
    """

    def __init__(self, judge_service: IJudgeService):
        """
        Initialize the service judge strategy.

        Args:
        ----
            judge_service: The judge service to use

        """
        self._judge_service = judge_service
        logger.info("ServiceJudgeStrategy initialized")

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

        """
        return await self._judge_service.judge(
            input_data=input_data,
            output_data=output_data,
            judgment_type=judgment_type,
            trace_id=trace_id,
        )

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
        return await self._judge_service.judge_with_rubric(
            input_data=input_data,
            output_data=output_data,
            rubric_name=rubric_name,
            trace_id=trace_id,
        )
