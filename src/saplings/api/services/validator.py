from __future__ import annotations

"""
Validator Service API module for Saplings.

This module provides the validator service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.providers.validator_service import (
    ValidatorService as _ValidatorService,
)


@stable
class ValidatorService(_ValidatorService):
    """
    Service for validating outputs.

    This service provides functionality for validating outputs, including
    checking for correctness, completeness, and other criteria.
    """


__all__ = [
    "ValidatorService",
]
