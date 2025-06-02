from __future__ import annotations

"""
Adapter for JudgeAgent to work with ValidatorService.

This module provides an adapter that translates between the ValidatorService's
expected parameters and the JudgeAgent's actual parameters.
"""

import logging
from typing import Any, Dict, List, Optional

from saplings.judge._internal.judge_agent import JudgeAgent, JudgeResult

logger = logging.getLogger(__name__)


class ValidatorServiceJudgeResult:
    """
    Wrapper for JudgeResult that provides the attributes expected by ValidatorService.

    This class adapts the JudgeResult class to the interface expected by ValidatorService.
    """

    def __init__(self, judge_result: JudgeResult):
        """
        Initialize the wrapper.

        Args:
        ----
            judge_result: The JudgeResult to wrap

        """
        self._judge_result = judge_result
        self.score = judge_result.overall_score
        self.feedback = judge_result.critique
        self.strengths: List[str] = []
        self.weaknesses: List[str] = []

        # Extract strengths and weaknesses from dimension scores
        for score in judge_result.dimension_scores:
            if score.score >= 0.7:  # Arbitrary threshold
                self.strengths.append(f"{score.dimension.value}: {score.explanation}")
            else:
                self.weaknesses.append(f"{score.dimension.value}: {score.explanation}")

        # Add suggestions to weaknesses if available
        if judge_result.suggestions:
            self.weaknesses.extend(judge_result.suggestions)


class JudgeAdapter:
    """
    Adapter for JudgeAgent to work with ValidatorService.

    This adapter translates between the ValidatorService's expected parameters
    and the JudgeAgent's actual parameters.
    """

    def __init__(self, judge: JudgeAgent):
        """
        Initialize the judge adapter.

        Args:
        ----
            judge: The JudgeAgent to adapt

        """
        self._judge_agent = (
            judge  # Store as a different attribute name to avoid method name conflict
        )

    async def judge(
        self,
        input_data: Dict[str, Any] | None = None,
        output_data: Any = None,
        judgment_type: Optional[str] = None,
        **kwargs,
    ) -> ValidatorServiceJudgeResult:
        """
        Judge an output using the JudgeAgent.

        This method adapts the parameters expected by ValidatorService to the
        parameters expected by JudgeAgent.

        Args:
        ----
            input_data: Input data dictionary containing prompt or task
            output_data: Output data to judge
            judgment_type: Type of judgment (unused in JudgeAgent)

        Returns:
        -------
            JudgeResult: Judgment result

        """
        # Extract prompt from input_data
        prompt = ""
        if input_data is not None:
            prompt = input_data.get("prompt", "")
            if not prompt and "task" in input_data:
                prompt = input_data.get("task", "")

        # Convert output_data to string if needed
        output = str(output_data) if not isinstance(output_data, str) else output_data

        # Acknowledge the judgment_type parameter to avoid unused parameter warning
        _ = judgment_type  # Not used in the base judge implementation
        _ = kwargs  # Acknowledge any additional parameters

        # Call the original judge with the correct parameter names
        # The judge method is a method of the JudgeAgent class, not a callable attribute
        judge_result = await self._judge_agent.judge(output=output, prompt=prompt)

        # Wrap the result with our adapter class
        return ValidatorServiceJudgeResult(judge_result)
