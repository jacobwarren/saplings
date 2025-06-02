from __future__ import annotations

"""
Service module for judge components.

This module provides service functionality for judges in the Saplings framework.
"""

from saplings.judge._internal.service.adapter import (
    JudgeAdapter,
    ValidatorServiceJudgeResult,
)
from saplings.judge._internal.service.judge_service import (
    DirectJudgeStrategy,
    IJudgeStrategy,
    JudgmentResult,
    ServiceJudgeStrategy,
)

__all__ = [
    "DirectJudgeStrategy",
    "IJudgeStrategy",
    "JudgeAdapter",
    "JudgmentResult",
    "ServiceJudgeStrategy",
    "ValidatorServiceJudgeResult",
]
