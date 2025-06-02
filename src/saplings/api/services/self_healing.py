from __future__ import annotations

"""
Self-Healing Service API module for Saplings.

This module provides the self-healing service implementation.
"""

from saplings.api.stability import beta
from saplings.services._internal.providers.self_healing_service import (
    SelfHealingService as _SelfHealingService,
)


@beta
class SelfHealingService(_SelfHealingService):
    """
    Service for self-healing.

    This service provides functionality for self-healing, including
    detecting and fixing issues, and learning from past mistakes.
    """


__all__ = [
    "SelfHealingService",
]
