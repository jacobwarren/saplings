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
from saplings.validator.validator import (
    RuntimeValidator,
    StaticValidator,
    ValidationResult,
    ValidationStatus,
    Validator,
)

__all__ = [
    "Validator",
    "StaticValidator",
    "RuntimeValidator",
    "ValidationResult",
    "ValidationStatus",
    "ValidatorConfig",
    "ValidatorType",
    "ValidatorRegistry",
    "get_validator_registry",
]
