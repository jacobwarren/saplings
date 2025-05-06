from __future__ import annotations

"""
Validator module for Saplings.

This module provides the validator functionality for Saplings, including:
- ValidatorRegistry for managing validators
- Base Validator interface
- StaticValidator and RuntimeValidator implementations
- Plugin discovery for validators
"""

from saplings.validator.config import ValidatorConfig, ValidatorType
from saplings.validator.registry import ValidatorRegistry, get_validator_registry
from saplings.validator.result import ValidationResult, ValidationStatus
from saplings.validator.validator import (
    RuntimeValidator,
    StaticValidator,
    Validator,
)

__all__ = [
    "RuntimeValidator",
    "StaticValidator",
    "ValidationResult",
    "ValidationStatus",
    "Validator",
    "ValidatorConfig",
    "ValidatorRegistry",
    "ValidatorType",
    "get_validator_registry",
]
