"""
Validator module for Saplings.

This module provides the validator functionality for Saplings, including:
- ValidatorRegistry for managing validators
- Base Validator interface
- StaticValidator and RuntimeValidator implementations
- Plugin discovery for validators
"""

from saplings.validator.config import ValidatorConfig, ValidatorType
from saplings.validator.validator import (
    Validator,
    StaticValidator,
    RuntimeValidator,
    ValidationResult,
    ValidationStatus,
)
from saplings.validator.registry import ValidatorRegistry, get_validator_registry

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
