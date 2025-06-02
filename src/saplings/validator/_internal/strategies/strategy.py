from __future__ import annotations

"""
Validation strategy module for Saplings.

This module provides strategy interfaces and implementations for validating outputs.
It follows the Strategy pattern to allow different validation strategies to be used
without changing the client code.
"""

import logging
from typing import Any, Dict, Optional, Protocol, TypeVar

from saplings.validator._internal.result import ValidationResult, ValidationStatus

# No need for IJudgeService import as we're using Any for judge_strategy

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IValidationStrategy(Protocol):
    """Interface for validation strategies."""

    async def validate(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        validation_type: str = "general",
        trace_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output quality.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: Validation results

        """
        ...

    async def validate_with_rubric(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        rubric_name: str,
        trace_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output using a specific rubric.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            rubric_name: Name of the rubric to use
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: Validation results

        """
        ...


class JudgeBasedValidationStrategy(IValidationStrategy):
    """
    Strategy that uses a JudgeService for validation.

    This strategy delegates to a JudgeService for validating outputs.
    """

    def __init__(self, judge_strategy: Any):
        """
        Initialize the judge-based validation strategy.

        Args:
        ----
            judge_strategy: The judge strategy to use

        """
        self._judge_strategy = judge_strategy
        logger.info("JudgeBasedValidationStrategy initialized")

    async def validate(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        validation_type: str = "general",
        trace_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output quality.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: Validation results

        """
        # Delegate to the judge strategy
        judgment = await self._judge_strategy.judge(
            input_data=input_data,
            output_data=output_data,
            judgment_type=validation_type,
            trace_id=trace_id,
        )

        # Convert judgment to validation result
        return ValidationResult(
            validator_id="judge_based_validator",
            status=ValidationStatus.PASSED if judgment.is_valid else ValidationStatus.FAILED,
            message=judgment.feedback,
            details=judgment.details,
        )

    async def validate_with_rubric(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        rubric_name: str,
        trace_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output using a specific rubric.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            rubric_name: Name of the rubric to use
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: Validation results

        """
        # Delegate to the judge strategy
        judgment = await self._judge_strategy.judge_with_rubric(
            input_data=input_data,
            output_data=output_data,
            rubric_name=rubric_name,
            trace_id=trace_id,
        )

        # Convert judgment to validation result
        return ValidationResult(
            validator_id="judge_based_validator",
            status=ValidationStatus.PASSED if judgment.is_valid else ValidationStatus.FAILED,
            message=judgment.feedback,
            details=judgment.details,
        )


class RuleBasedValidationStrategy(IValidationStrategy):
    """
    Strategy that uses predefined rules for validation.

    This strategy uses a set of rules to validate outputs without
    requiring a judge or LLM.
    """

    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        """
        Initialize the rule-based validation strategy.

        Args:
        ----
            rules: Optional dictionary of validation rules

        """
        self._rules = rules or {}
        logger.info("RuleBasedValidationStrategy initialized")

    async def validate(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        validation_type: str = "general",
        trace_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output quality.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: Validation results

        """
        # Get rules for the validation type
        type_rules = self._rules.get(validation_type, {})
        if not type_rules:
            logger.warning(f"No rules found for validation type: {validation_type}")
            return ValidationResult(
                validator_id="rule_based_validator",
                status=ValidationStatus.WARNING,
                message="No validation rules defined",
                details={"warning": f"No rules found for validation type: {validation_type}"},
            )

        # Apply rules
        is_valid = True
        feedback = "Validation passed"
        details = {}

        # TODO: Implement rule-based validation logic
        # This is a placeholder for actual rule-based validation

        return ValidationResult(
            validator_id="rule_based_validator",
            status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
            message=feedback,
            details=details,
        )

    async def validate_with_rubric(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        rubric_name: str,
        trace_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output using a specific rubric.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            rubric_name: Name of the rubric to use
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: Validation results

        """
        # Get rules for the rubric
        rubric_rules = self._rules.get(rubric_name, {})
        if not rubric_rules:
            logger.warning(f"No rules found for rubric: {rubric_name}")
            return ValidationResult(
                validator_id="rule_based_validator",
                status=ValidationStatus.WARNING,
                message="No validation rules defined",
                details={"warning": f"No rules found for rubric: {rubric_name}"},
            )

        # Apply rules
        is_valid = True
        feedback = "Validation passed"
        details = {}

        # TODO: Implement rule-based validation logic
        # This is a placeholder for actual rule-based validation

        return ValidationResult(
            validator_id="rule_based_validator",
            status=ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED,
            message=feedback,
            details=details,
        )
