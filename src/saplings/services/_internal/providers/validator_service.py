from __future__ import annotations

"""
saplings.services.validator_service.
=================================

Encapsulates validation functionality:
- Output validation
- Validator registry management
- Event-based communication with other services
"""


import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from saplings.api.core.interfaces import IJudgeService, IValidatorService
from saplings.core._internal.exceptions import (
    ConfigurationError,
    OperationCancelledError,
    OperationTimeoutError,
    ValidationError,
)
from saplings.core._internal.validation.validation import validate_not_empty, validate_required
from saplings.core.events import CoreEvent, CoreEventType, get_event_bus
from saplings.core.mediator import ServiceRequest, ServiceResponse, get_service_mediator
from saplings.core.resilience import DEFAULT_TIMEOUT
from saplings.core.resilience_patterns import ResiliencePatterns

# Use TYPE_CHECKING for type hints to avoid circular imports
if TYPE_CHECKING:
    from saplings.api.validator import (
        IValidationStrategy,
        ValidatorRegistry,
    )
    from saplings.core._internal.model_adapter import LLM

# Optional dependency (monitoring)
try:
    from saplings.monitoring._internal import TraceManager
except ModuleNotFoundError:  # pragma: no cover
    TraceManager = None  # type: ignore

logger = logging.getLogger(__name__)


class ValidatorService(IValidatorService):
    """Service that manages validation of outputs."""

    def __init__(
        self,
        model: Optional["LLM"] = None,
        trace_manager: Any = None,
        judge_service: Optional[IJudgeService] = None,
        validation_strategy: Optional[IValidationStrategy] = None,
        validator_registry: Optional["ValidatorRegistry"] = None,
    ) -> None:
        """
        Initialize the validator service.

        Args:
        ----
            model: Optional model to use for validation
            trace_manager: Optional trace manager for monitoring
            judge_service: Optional judge service for delegating validation
            validation_strategy: Optional validation strategy to use
            validator_registry: Optional validator registry to use

        Raises:
        ------
            ValidationError: If validator initialization fails

        """
        try:
            # Initialize core dependencies
            self._trace_manager = trace_manager
            self._model = model
            self._judge_service = judge_service
            self._validation_history: List[Dict[str, Any]] = []
            self._event_bus = get_event_bus()
            self._mediator = get_service_mediator()
            self._initialized = False
            self._validator_registry = validator_registry
            self._validation_strategy = validation_strategy
            self.validator = None

            # Register for events and mediator handlers
            self._register_event_handlers()
            self._register_mediator_handlers()

            logger.info("ValidatorService initialized with lazy loading")
        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    f"Failed to initialize validator service: {e!s}",
                    validation_type="initialization",
                    cause=e,
                )
            raise

    def _init_validation_strategy(self, validation_strategy: Optional[Any]) -> None:
        """
        Initialize the validation strategy.

        Args:
        ----
            validation_strategy: Optional validation strategy to use

        Raises:
        ------
            ValidationError: If validation strategy initialization fails

        """
        # Set validation strategy
        self._validation_strategy = validation_strategy

        # Create a strategy if none is provided
        if self._validation_strategy is None:
            # Import validation strategies lazily to avoid circular imports
            from saplings.api.validator import (
                JudgeBasedValidationStrategy,
                RuleBasedValidationStrategy,
            )

            if self._judge_service:
                # Create a judge-based strategy if judge_service is provided
                from saplings.judge._internal.strategy import ServiceJudgeStrategy

                judge_strategy = ServiceJudgeStrategy(self._judge_service)
                self._validation_strategy = JudgeBasedValidationStrategy(judge_strategy)
                logger.debug("Created judge-based validation strategy")
            else:
                # Create a rule-based strategy as fallback
                self._validation_strategy = RuleBasedValidationStrategy()
                logger.debug("Created rule-based validation strategy")

        # Ensure validation strategy is not None
        if self._validation_strategy is None:
            raise ValidationError(
                "Failed to initialize validation strategy",
                validation_type="initialization",
            )

    def _register_event_handlers(self) -> None:
        """Register event handlers for cross-service communication."""
        # Subscribe to judge events
        self._event_bus.subscribe(
            CoreEventType.JUDGE_COMPLETED,
            self._handle_judge_completed,
        )

    def _register_mediator_handlers(self) -> None:
        """Register handlers with the service mediator."""
        self._mediator.register_handler(
            "validate",
            self._handle_validate_request,
        )
        self._mediator.register_handler(
            "validate_with_rubric",
            self._handle_validate_with_rubric_request,
        )
        logger.debug("Registered mediator handlers")

    def _handle_validate_request(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle validation requests from the mediator.

        Args:
        ----
            request: The request to handle

        Returns:
        -------
            ServiceResponse: The response

        """
        try:
            # Extract request data
            input_data = request.data.get("input_data", {})
            output_data = request.data.get("output_data", "")
            validation_type = request.data.get("validation_type", "general")
            trace_id = request.data.get("trace_id")

            # Create a task to run the validation
            import asyncio

            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.validate(
                    input_data=input_data,
                    output_data=output_data,
                    validation_type=validation_type,
                    trace_id=trace_id,
                )
            )

            return ServiceResponse(
                success=True,
                data={"result": result},
            )
        except Exception as e:
            logger.exception(f"Error handling validate request: {e}")
            return ServiceResponse(
                success=False,
                data={},
                error=str(e),
            )

    def _handle_validate_with_rubric_request(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle validate with rubric requests from the mediator.

        Args:
        ----
            request: The request to handle

        Returns:
        -------
            ServiceResponse: The response

        """
        try:
            # Extract request data
            input_data = request.data.get("input_data", {})
            output_data = request.data.get("output_data", "")
            rubric_name = request.data.get("rubric_name", "")
            trace_id = request.data.get("trace_id")

            # Create a task to run the validation
            import asyncio

            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.validate_with_rubric(
                    input_data=input_data,
                    output_data=output_data,
                    rubric_name=rubric_name,
                    trace_id=trace_id,
                )
            )

            return ServiceResponse(
                success=True,
                data={"result": result},
            )
        except Exception as e:
            logger.exception(f"Error handling validate with rubric request: {e}")
            return ServiceResponse(
                success=False,
                data={},
                error=str(e),
            )

    def _handle_judge_completed(self, event: CoreEvent) -> None:
        """
        Handle judge completed events.

        Args:
        ----
            event: The event to handle

        """
        logger.debug(f"Received judge completed event: {event}")
        # We could update internal state based on judge events if needed

    def _ensure_initialized(self) -> None:
        """
        Ensure the validator service is fully initialized.

        This method lazily initializes the validator service components when they are first needed.

        Raises
        ------
            ValidationError: If initialization fails

        """
        if self._initialized:
            return

        try:
            # Initialize validator registry if not provided
            if self._validator_registry is None:
                try:
                    # Import lazily to avoid circular imports
                    from saplings.api.validator import get_validator_registry

                    self._validator_registry = get_validator_registry()
                except Exception as e:
                    # Create a new validator registry if we can't get one from the container
                    # Import lazily to avoid circular imports
                    from saplings.api.validator import ValidatorRegistry

                    self._validator_registry = ValidatorRegistry()
                    logger.warning(
                        f"Failed to get validator registry from container, created a new one: {e}"
                    )

            # Initialize validation strategy if not provided
            if self._validation_strategy is None:
                self._init_validation_strategy(None)

            # Mark as initialized
            self._initialized = True
            logger.debug("ValidatorService fully initialized on-demand")
        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    f"Failed to initialize validator service: {e!s}",
                    validation_type="initialization",
                    cause=e,
                )
            raise

    def _init_validator(self) -> None:
        """
        Initialize the validator to ensure it's never None.

        Raises
        ------
            ValidationError: If validator initialization fails

        """
        # Ensure service is initialized
        self._ensure_initialized()

        # Default to getting the first validator from the registry if available
        if not hasattr(self, "validator") or self.validator is None:
            try:
                # Get the default validator (execution validator)
                self.validator = self._validator_registry.get_validator("execution")
                logger.debug("Initialized default validator")
            except ValueError as e:
                try:
                    # If execution validator is not found, create a fallback validator
                    # Import lazily to avoid circular imports
                    from saplings.api.validator import LengthValidator

                    self.validator = LengthValidator()
                    logger.warning(
                        "Execution validator not found, using LengthValidator as fallback"
                    )
                except Exception as inner_e:
                    raise ValidationError(
                        "Failed to initialize validator",
                        validation_type="initialization",
                        cause=inner_e,
                    ) from e

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    async def validate(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        validation_type: str = "execution",
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
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
            Dict[str, Any]: Validation result

        Raises:
        ------
            ValidationError: If validation fails
            ConfigurationError: If required parameters are missing or invalid
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        # Validate required parameters
        try:
            validate_required(input_data, "input_data")
            validate_required(output_data, "output_data")
            validate_not_empty(validation_type, "validation_type")
        except Exception as e:
            raise ConfigurationError(
                f"Invalid validation parameters: {e!s}",
                config_key="validation_type",
                config_value=validation_type,
            ) from e
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

        # Publish validation requested event
        self._event_bus.publish(
            CoreEvent(
                event_type=CoreEventType.VALIDATION_REQUESTED,
                data={
                    "validation_type": validation_type,
                    "input_length": len(str(input_data)),
                    "output_length": len(str(output_data)),
                },
                source="ValidatorService",
                trace_id=trace_id,
            )
        )

        try:
            # Ensure service and validator are initialized
            self._ensure_initialized()
            self._init_validator()

            # Get the appropriate validator (not used directly but ensures it exists)
            _ = self._validator_registry.get_validator(validation_type)

            # Define validation function
            async def _validate():
                # Use the validation strategy (already validated it's not None in __init__)
                if self._validation_strategy is None:
                    raise ValidationError(
                        "Validation strategy is not initialized",
                        validation_type=validation_type,
                    )

                result = await self._validation_strategy.validate(
                    input_data=input_data,
                    output_data=output_data,
                    validation_type=validation_type,
                    trace_id=trace_id,
                )

                # Add to history
                self._validation_history.append(
                    {
                        "input": input_data,
                        "output": output_data,
                        "validation_type": validation_type,
                        "result": result,
                    }
                )

                return {
                    "is_valid": result.is_valid,
                    "score": result.score,
                    "feedback": result.feedback,
                    "details": result.details or {},
                }

            # Execute with timeout using resilience pattern
            result = await ResiliencePatterns.with_timeout(
                _validate(), timeout=timeout, operation_name="validate"
            )

            # Publish validation completed event
            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.VALIDATION_COMPLETED,
                    data={
                        "validation_type": validation_type,
                        "result": result,
                    },
                    source="ValidatorService",
                    trace_id=trace_id,
                )
            )

            return result
        except Exception as e:
            # Publish validation failed event
            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.VALIDATION_FAILED,
                    data={"error": str(e)},
                    source="ValidatorService",
                    trace_id=trace_id,
                )
            )
            logger.exception(f"Error during validation: {e}")

            # Wrap exception if it's not already a ValidationError
            if not isinstance(e, (ValidationError, OperationTimeoutError, OperationCancelledError)):
                raise ValidationError(
                    f"Validation failed: {e!s}",
                    validation_type=validation_type,
                    input_data=input_data,
                    output_data=output_data,
                    cause=e,
                ) from e
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    async def judge_output(
        self,
        input_data: Dict[str, Any],
        output_data: Any,
        judgment_type: str = "general",
        trace_id: Optional[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Judge output quality by delegating to the JudgeService via events.

        Args:
        ----
            input_data: Input data that produced the output
            output_data: Output data to judge
            judgment_type: Type of judgment to perform
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Dict[str, Any]: Judgment results

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
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
            # Publish judge requested event
            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.JUDGE_REQUESTED,
                    data={
                        "judgment_type": judgment_type,
                        "input_length": len(str(input_data)),
                        "output_length": len(str(output_data)),
                    },
                    source="ValidatorService",
                    trace_id=trace_id,
                )
            )

            # Ensure service is initialized
            self._ensure_initialized()

            # Use the validation strategy directly
            try:
                # Define validation function
                async def _judge():
                    if self._validation_strategy is None:
                        raise ValidationError(
                            "Validation strategy is not initialized",
                            validation_type=judgment_type,
                        )

                    result = await self._validation_strategy.validate(
                        input_data=input_data,
                        output_data=output_data,
                        validation_type=judgment_type,
                        trace_id=trace_id,
                    )
                    return {
                        "score": result.score,
                        "feedback": result.feedback,
                        "is_valid": result.is_valid,
                        "details": result.details or {},
                    }

                # Execute with timeout using resilience pattern
                return await ResiliencePatterns.with_timeout(
                    _judge(), timeout=timeout, operation_name="judge"
                )
            except Exception as e:
                logger.warning(f"Failed to use validation strategy: {e}")
                # If we have a judge service, use it directly
                if self._judge_service:
                    logger.info("Using judge service directly")
                    result = await self._judge_service.judge(
                        input_data=input_data,
                        output_data=output_data,
                        judgment_type=judgment_type,
                        trace_id=trace_id,
                    )
                    return {
                        "score": result.score,
                        "feedback": result.feedback,
                        "strengths": result.strengths if hasattr(result, "strengths") else [],
                        "weaknesses": result.weaknesses if hasattr(result, "weaknesses") else [],
                    }
                else:
                    # Fallback to direct validation
                    logger.info("Falling back to direct validation")
                    return await self.validate(
                        input_data=input_data,
                        output_data=output_data,
                        validation_type="general",
                        trace_id=trace_id,
                        timeout=timeout,
                    )
        except Exception as e:
            logger.exception(f"Error during judgment: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """
        Get validation history.

        Returns
        -------
            List[Dict[str, Any]]: History of validation operations

        """
        return self._validation_history

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
        # Ensure service is initialized
        self._ensure_initialized()

        # Use the validation strategy directly
        try:
            if self._validation_strategy is None:
                raise ValidationError(
                    "Validation strategy is not initialized",
                    validation_type="rubric",
                )

            result = await self._validation_strategy.validate_with_rubric(
                input_data=input_data,
                output_data=output_data,
                rubric_name=rubric_name,
                trace_id=trace_id,
            )
            return {
                "score": result.score,
                "feedback": result.feedback,
                "is_valid": result.is_valid,
                "details": result.details or {},
            }
        except Exception as e:
            logger.warning(f"Failed to use validation strategy: {e}")
            # If we have a judge service, use it directly
            if self._judge_service:
                logger.info("Using judge service directly")
                result = await self._judge_service.judge_with_rubric(
                    input_data=input_data,
                    output_data=output_data,
                    rubric_name=rubric_name,
                    trace_id=trace_id,
                )
                return {
                    "score": result.score,
                    "feedback": result.feedback,
                    "strengths": result.strengths if hasattr(result, "strengths") else [],
                    "weaknesses": result.weaknesses if hasattr(result, "weaknesses") else [],
                }
            else:
                # Fallback to regular validation
                logger.info("Falling back to direct validation")
                return await self.validate(
                    input_data=input_data,
                    output_data=output_data,
                    validation_type="general",
                    trace_id=trace_id,
                )
