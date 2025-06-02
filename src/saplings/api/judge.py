from __future__ import annotations

"""
Judge API module for Saplings.

This module provides the public API for judge components, including:
- Output scoring and verification
- Structured critique generation
- Rubric-based evaluation
- Budget enforcement
- Judge strategies
"""

from saplings.api.stability import beta
from saplings.judge._internal.config import CritiqueFormat as _CritiqueFormat
from saplings.judge._internal.config import JudgeConfig as _JudgeConfig
from saplings.judge._internal.config import Rubric as _Rubric
from saplings.judge._internal.config import RubricItem as _RubricItem
from saplings.judge._internal.config import ScoringDimension as _ScoringDimension
from saplings.judge._internal.judge import JudgeAgent as _JudgeAgent
from saplings.judge._internal.judge import JudgeResult as _JudgeResult
from saplings.judge._internal.service import DirectJudgeStrategy as _DirectJudgeStrategy
from saplings.judge._internal.service import ServiceJudgeStrategy as _ServiceJudgeStrategy

# Re-export the enum directly
CritiqueFormat = _CritiqueFormat


@beta
class JudgeConfig(_JudgeConfig):
    """
    Configuration for the judge agent.

    This class defines the configuration options for the judge agent,
    including the rubric, critique format, and scoring dimensions.
    """


@beta
class Rubric(_Rubric):
    """
    Rubric for evaluating outputs.

    A rubric is a collection of rubric items that define the criteria
    for evaluating outputs. Each rubric item has a name, description,
    and weight.
    """


@beta
class RubricItem(_RubricItem):
    """
    Item in a rubric for evaluating outputs.

    A rubric item defines a specific criterion for evaluating outputs,
    with a name, description, and weight.
    """


# Re-export the enum directly
ScoringDimension = _ScoringDimension


@beta
class JudgeAgent(_JudgeAgent):
    """
    Agent for judging outputs.

    The judge agent evaluates outputs based on a rubric, generating
    scores and critiques for each output.
    """


@beta
class JudgeResult(_JudgeResult):
    """
    Result of a judge evaluation.

    Contains the scores, critique, and overall evaluation of an output
    based on the rubric.
    """


@beta
class DirectJudgeStrategy(_DirectJudgeStrategy):
    """
    Strategy that uses a JudgeAgent directly.

    This strategy initializes and uses a JudgeAgent directly without
    going through the JudgeService.
    """


@beta
class ServiceJudgeStrategy(_ServiceJudgeStrategy):
    """
    Strategy that uses the JudgeService.

    This strategy delegates to the JudgeService for judging outputs.
    """


__all__ = [
    # Enums
    "CritiqueFormat",
    "ScoringDimension",
    # Configuration classes
    "JudgeConfig",
    "Rubric",
    "RubricItem",
    # Core classes
    "JudgeAgent",
    "JudgeResult",
    # Strategies
    "DirectJudgeStrategy",
    "ServiceJudgeStrategy",
]
