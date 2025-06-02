from __future__ import annotations

"""
Internal module for judge components.

This module provides the implementation of judge components for the Saplings framework.
"""

# Import from individual modules
# Import from subdirectories
from saplings.judge._internal.config import (
    CritiqueFormat,
    JudgeConfig,
    Rubric,
    RubricItem,
    ScoringDimension,
)
from saplings.judge._internal.judge import DimensionScore, JudgeAgent, JudgeResult
from saplings.judge._internal.rubrics import (
    RubricLoader,
    RubricTemplate,
    RubricValidator,
)
from saplings.judge._internal.service import (
    DirectJudgeStrategy,
    IJudgeStrategy,
    JudgeAdapter,
    JudgmentResult,
    ServiceJudgeStrategy,
    ValidatorServiceJudgeResult,
)

__all__ = [
    # Core judge
    "DimensionScore",
    "JudgeAgent",
    "JudgeResult",
    # Configuration
    "CritiqueFormat",
    "JudgeConfig",
    "Rubric",
    "RubricItem",
    "ScoringDimension",
    # Rubrics
    "RubricLoader",
    "RubricTemplate",
    "RubricValidator",
    # Service
    "DirectJudgeStrategy",
    "IJudgeStrategy",
    "JudgeAdapter",
    "JudgmentResult",
    "ServiceJudgeStrategy",
    "ValidatorServiceJudgeResult",
]
