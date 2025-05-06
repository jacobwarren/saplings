from __future__ import annotations

"""
saplings.services.validator_service.
=================================

Encapsulates validation functionality:
- Output validation
- Judge integration
- Validator registry management
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.core.resilience import DEFAULT_TIMEOUT, run_in_executor, with_timeout
from saplings.validator.registry import get_validator_registry

if TYPE_CHECKING:
    from saplings.core.model_adapter import LLM
    from saplings.monitoring.trace import TraceManager

# Optional dependency (monitoring)
try:
    from saplings.monitoring.trace import TraceManager  # noqa
except ModuleNotFoundError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


class ValidatorService:
    """Service that manages validation of outputs."""

    def __init__(
        self,
        model: "LLM",
        trace_manager: Any = None,
        model_service=None,  # Added for compatibility with tests
    ) -> None:
        self._trace_manager = trace_manager
        self._model = model

        # Initialize validator registry
        # Initialize validator registry
        self.validator_registry = get_validator_registry()

        # Initialize validator to ensure it's never None
        self._init_validator()

        # Initialize judge (can be set later via set_judge)
        self.judge: Any = None
        # Initialize judge with a default implementation or placeholder
        self._init_judge()

        logger.info("ValidatorService initialized")

    def _init_validator(self):
        """Initialize the validator to ensure it's never None."""
        # Default to getting the first validator from the registry if available
        if not hasattr(self, "validator") or self.validator is None:
            # Get the default validator (execution validator)
            self.validator = self.validator_registry.get_validator("execution")
            logger.debug("Initialized default validator")

    def _init_judge(self):
        """Initialize the judge to ensure it's never None."""
        # Judge will be set externally via set_judge method
        # This method exists to be overridden in subclasses if needed
        if not hasattr(self, "judge") or self.judge is None:
            # Judge requires external initialization, but we can set up a stub
            # to prevent AttributeErrors
            self.judge: Any = None
            logger.debug("Judge initialized as None (will be set later)")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    async def validate(
        self,
        input_data: dict[str, Any],
        output_data: Any,
        validation_type: str = "execution",
        trace_id: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
    ):
        """
        Validate an output against an input.

        Args:
        ----
            input_data: Input data (prompt, context, etc.)
            output_data: Output data to validate
            validation_type: Type of validation to perform
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Validation result

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ValidatorService.validate",
                trace_id=trace_id,
                attributes={
                    "component": "validator_service",
                    "validation_type": validation_type,
                },
            )

        try:
            # Ensure validator is initialized
            self._init_validator()

            # Get the appropriate validator
            validator = self.validator_registry.get_validator(validation_type)

            # Define validation function
            async def _validate():
                return await validator.validate(
                    input_data=input_data,
                    output_data=output_data,
                    model=self._model,
                )

            # Execute with timeout
            return await with_timeout(_validate(), timeout=timeout, operation_name="validate")
        except Exception as e:
            logger.exception(f"Error during validation: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    async def judge_output(
        self,
        input_data: dict[str, Any],
        output_data: Any,
        judgment_type: str = "general",
        trace_id: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """
        Judge an output using the JudgeAgent.

        Args:
        ----
            input_data: Input data
            output_data: Output data to judge
            judgment_type: Type of judgment
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Judgment result

        Raises:
        ------
            ValueError: If the judge is not initialized
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        # Ensure judge is initialized
        if self.judge is None:
            msg = "Judge is not initialized. Call set_judge() before using judge_output()."
            raise ValueError(msg)

        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="ValidatorService.judge_output",
                trace_id=trace_id,
                attributes={
                    "component": "validator_service",
                    "judgment_type": judgment_type,
                },
            )

        try:
            # Define judging function
            async def _judge_output():
                # We've already checked that self.judge is not None at the beginning of the method
                # but let's add an additional check here for type safety
                if not self.judge:
                    msg = "Judge is not initialized. Call set_judge() before using judge_output()."
                    raise ValueError(msg)

                judgment = await self.judge.judge(
                    input_data=input_data,
                    output_data=output_data,
                    judgment_type=judgment_type,
                )

                logger.info("Judged output with score: %s", judgment.score)

                return {
                    "score": judgment.score,
                    "feedback": judgment.feedback,
                    "strengths": judgment.strengths,
                    "weaknesses": judgment.weaknesses,
                }

            # Execute with timeout
            return await with_timeout(
                _judge_output(), timeout=timeout, operation_name="judge_output"
            )
        except Exception as e:
            logger.exception(f"Error during judgment: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    async def get_validation_history(
        self, timeout: float | None = DEFAULT_TIMEOUT
    ) -> list[dict[str, Any]]:
        """
        Get validation history.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Dict[str, Any]]: Validation history

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """

        # Define function to get history (placeholder implementation)
        def _get_history():
            # This method might require additional implementation
            # depending on how validation history is tracked
            return []

        # Run in executor with timeout
        return await run_in_executor(_get_history, timeout=timeout)

    async def set_judge(self, judge, timeout: float | None = DEFAULT_TIMEOUT) -> None:
        """
        Set the judge agent for validation.

        Args:
        ----
            judge: The judge agent to set
            timeout: Optional timeout in seconds

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """

        # Define function to set judge
        def _set_judge(j):
            self.judge = j
            logger.info("Judge set for ValidatorService")
            return True

        # Run in executor with timeout
        await run_in_executor(_set_judge, judge, timeout=timeout)
