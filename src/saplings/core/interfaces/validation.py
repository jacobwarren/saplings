from __future__ import annotations

"""
Validator service interface for Saplings.

This module defines the interface for validation operations that verify
output quality and correctness. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any


class IValidatorService(ABC):
    """Interface for validation operations."""

    @abstractmethod
    async def validate(
        self,
        input_data: dict[str, Any],
        output_data: Any,
        validation_type: str = "general",
        trace_id: str | None = None,
    ) -> dict[str, Any]:
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
        input_data: dict[str, Any],
        output_data: Any,
        judgment_type: str = "general",
        trace_id: str | None = None,
    ) -> dict[str, Any]:
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
    def set_judge(self, judge: Any) -> None:
        """
        Set the judge for validation.

        Args:
        ----
            judge: Judge instance

        """

    @abstractmethod
    def get_validation_history(self):
        """
        Get validation history.

        Returns
        -------
            List[Dict[str, Any]]: History of validation operations

        """

    @abstractmethod
    async def create_rubric(
        self, name: str, description: str, criteria: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Create a validation rubric.

        Args:
        ----
            name: Rubric name
            description: Rubric description
            criteria: List of criteria definitions

        Returns:
        -------
            Dict[str, Any]: The created rubric

        """

    @abstractmethod
    async def validate_with_rubric(
        self,
        input_data: dict[str, Any],
        output_data: Any,
        rubric_name: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
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
