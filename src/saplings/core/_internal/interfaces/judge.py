from __future__ import annotations

"""
Judge service interface for Saplings.

This module defines the interface for judge operations that evaluate
output quality. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class JudgmentContext:
    """Standard context for judgment operations."""

    input_data: Dict[str, Any]
    output_data: Any
    judgment_type: str = "general"
    trace_id: Optional[str] = None
    timeout: Optional[float] = None


@dataclass
class JudgmentResult:
    """Standard result for judgment operations."""

    score: float
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    details: Optional[Dict[str, Any]] = None


@dataclass
class JudgeConfig:
    """Configuration for judge operations."""

    criteria: List[str]
    scoring_scale: int = 10
    detailed_feedback: bool = True
    model_name: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class JudgeResult:
    """Result of a judge operation."""

    score: float
    feedback: str
    criteria_scores: Dict[str, float]
    detailed_feedback: Optional[Dict[str, str]] = None


class IJudgeService(ABC):
    """Interface for judge operations."""

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def get_judgment_history(self) -> List[Dict[str, Any]]:
        """
        Get judgment history.

        Returns
        -------
            List[Dict[str, Any]]: History of judgment operations

        """

    @abstractmethod
    async def evaluate(
        self,
        input_text: str,
        output_text: str,
        config: Optional[JudgeConfig] = None,
        trace_id: Optional[str] = None,
    ) -> JudgeResult:
        """
        Evaluate output against input using specified criteria.

        Args:
        ----
            input_text: Input text
            output_text: Output text to evaluate
            config: Optional judge configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            JudgeResult: Evaluation results

        """
