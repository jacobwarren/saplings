from __future__ import annotations

"""
Validator service interface for Saplings.

This module defines the interface for validation operations that verify
output quality and correctness. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationContext:
    """Standard context for validation operations."""

    input_data: Dict[str, Any]
    output_data: Any
    validation_type: str = "general"
    trace_id: Optional[str] = None
    timeout: Optional[float] = None


@dataclass
class ValidationResult:
    """Standard result for validation operations."""

    is_valid: bool
    score: float
    feedback: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""

    validation_type: str = "general"
    criteria: List[str] = field(default_factory=list)
    model_name: Optional[str] = None
    provider: Optional[str] = None
    threshold: float = 0.7


class IValidatorService(ABC):
    """Interface for validation operations."""

    @abstractmethod
    async def validate(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        validation_type: str = "general",
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate output against input data.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: Validation results

        """

    @abstractmethod
    async def judge_output(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        judgment_type: str = "general",
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Judge output quality.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to judge
            judgment_type: Type of judgment to perform
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: Judgment results

        """

    @abstractmethod
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """
        Get validation history.

        Returns
        -------
            List[Dict[str, Any]]: History of validation operations

        """

    @abstractmethod
    async def validate_with_rubric(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        rubric_name: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            Dict[str, Any]: Validation results

        """

    @abstractmethod
    async def validate_with_config(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        config: Optional[ValidationConfig] = None,
        trace_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output with configuration.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to validate
            config: Optional validation configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            ValidationResult: Validation results

        """
