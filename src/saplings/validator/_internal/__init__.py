from __future__ import annotations

"""
Internal module for validator components.

This module provides the implementation of validator components for the Saplings framework.
"""

# Import from individual modules
from saplings.validator._internal.config import ValidatorConfig, ValidatorType
from saplings.validator._internal.implementations import (
    ExecutionValidator,
    KeywordValidator,
    LengthValidator,
    PiiValidator,
    ProfanityValidator,
    SentimentValidator,
)

# Import from subdirectories
from saplings.validator._internal.registry import (
    ValidatorRegistry,
    get_validator_registry,
)
from saplings.validator._internal.result import ValidationResult, ValidationStatus
from saplings.validator._internal.strategies import (
    IValidationStrategy as ValidationStrategy,
)
from saplings.validator._internal.strategies import (
    JudgeBasedValidationStrategy,
    RuleBasedValidationStrategy,
)
from saplings.validator._internal.validator import RuntimeValidator, StaticValidator, Validator

__all__ = [
    # Core components
    "RuntimeValidator",
    "StaticValidator",
    "ValidationResult",
    "ValidationStatus",
    "Validator",
    "ValidatorConfig",
    "ValidatorType",
    # Registry
    "ValidatorRegistry",
    "get_validator_registry",
    # Strategies
    "ValidationStrategy",
    "JudgeBasedValidationStrategy",
    "RuleBasedValidationStrategy",
    # Implementations
    "ExecutionValidator",
    "KeywordValidator",
    "LengthValidator",
    "PiiValidator",
    "ProfanityValidator",
    "SentimentValidator",
]
