from __future__ import annotations

"""
Judge Service API module for Saplings.

This module provides the judge service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.providers.judge_service import JudgeService as _JudgeService


@stable
class JudgeService(_JudgeService):
    """
    Service for judging outputs.

    This service provides functionality for judging outputs, including
    evaluating quality, correctness, and other criteria.
    """


__all__ = [
    "JudgeService",
]
