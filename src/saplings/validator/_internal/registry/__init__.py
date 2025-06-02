from __future__ import annotations

"""
Registry module for validator components.

This module provides registry functionality for validators in the Saplings framework.
"""

from saplings.validator._internal.registry.validator_registry import (
    ValidatorRegistry,
    get_validator_registry,
)

__all__ = [
    "ValidatorRegistry",
    "get_validator_registry",
]
