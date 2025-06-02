from __future__ import annotations

"""
Exception classes for Saplings.

This module provides custom exception classes for Saplings to enable
more granular error handling and preserve stack traces.

The exception hierarchy is designed to be comprehensive and consistent,
allowing for precise error handling throughout the codebase.
"""


from typing import Any, Optional


class SaplingsError(Exception):
    """Base exception for all Saplings errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """
        Initialize the exception.

        Args:
        ----
            message: Error message
            cause: Original exception that caused this one

        """
        self.cause = cause
        self.error_info: dict[str, Any] = {}

        # If we have a cause, include its message and type in our message
        if cause:
            # Include cause information in the message
            cause_msg = f" (caused by {cause.__class__.__name__}: {cause!s})"
            full_message = message + cause_msg
        else:
            full_message = message

        super().__init__(full_message)

    def with_info(self, **kwargs) -> "SaplingsError":
        """
        Add additional error information to the exception.

        Args:
        ----
            **kwargs: Key-value pairs to add to error_info

        Returns:
        -------
            Self for method chaining

        """
        self.error_info.update(kwargs)
        return self

    @classmethod
    def from_exception(cls, exception: Exception, message: Optional[str] = None) -> "SaplingsError":
        """
        Create a SaplingsError from another exception.

        Args:
        ----
            exception: The exception to wrap
            message: Optional message to use instead of the exception's message

        Returns:
        -------
            A new SaplingsError instance

        """
        msg = message or str(exception)
        return cls(msg, cause=exception)


class ModelError(SaplingsError):
    """Base exception for model-related errors."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model error.

        Args:
        ----
            message: Error message
            cause: Original exception that caused this one
            model_name: Name of the model that caused the error
            provider: Provider of the model that caused the error
            prompt: Prompt that caused the error
            **kwargs: Additional error information

        """
        super().__init__(message, cause=cause)
        self.model_name = model_name
        self.provider = provider
        self.prompt = prompt
        self.with_info(model_name=model_name, provider=provider, prompt=prompt, **kwargs)


class AdapterError(ModelError):
    """Exception for model adapter errors."""


class ProviderError(ModelError):
    """Exception for model provider errors."""


class APIError(ProviderError):
    """Exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None, **kwargs) -> None:
        """
        Initialize the API error.

        Args:
        ----
            message: Error message
            status_code: HTTP status code
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.status_code = status_code
        self.with_info(status_code=status_code, **kwargs)


class RateLimitError(APIError):
    """Exception for rate limit errors."""


class AuthenticationError(APIError):
    """Exception for authentication errors."""


class QuotaExceededError(APIError):
    """Exception for quota exceeded errors."""


class InvalidRequestError(APIError):
    """Exception for invalid request errors."""


class ServiceError(SaplingsError):
    """Base exception for service-related errors."""


class ConfigurationError(SaplingsError):
    """Exception for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the configuration error.

        Args:
        ----
            message: Error message
            config_key: The configuration key that caused the error
            config_value: The invalid configuration value
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value
        self.with_info(config_key=config_key, config_value=config_value, **kwargs)


class MissingDependencyError(SaplingsError):
    """Exception for missing dependencies."""

    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        installation_instructions: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the missing dependency error.

        Args:
        ----
            message: Error message
            dependency_name: Name of the missing dependency
            installation_instructions: Instructions for installing the dependency
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.dependency_name = dependency_name
        self.installation_instructions = installation_instructions
        self.with_info(
            dependency_name=dependency_name,
            installation_instructions=installation_instructions,
            **kwargs,
        )


class InitializationError(ServiceError):
    """Exception for service initialization errors."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        missing_dependencies: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the initialization error.

        Args:
        ----
            message: Error message
            service_name: Name of the service that failed to initialize
            missing_dependencies: List of missing dependencies
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.service_name = service_name
        self.missing_dependencies = missing_dependencies or []
        self.with_info(
            service_name=service_name, missing_dependencies=missing_dependencies, **kwargs
        )


class MemoryError(ServiceError):
    """Exception for memory management errors."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        component: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the memory error.

        Args:
        ----
            message: Error message
            cause: Original exception that caused this one
            component: The memory component that caused the error
            **kwargs: Additional error information

        """
        super().__init__(message, cause=cause)
        self.component = component
        self.with_info(component=component, **kwargs)


class IndexError(MemoryError):
    """Exception for indexing errors."""


class RetrievalError(ServiceError):
    """Exception for retrieval errors."""


class ToolError(ServiceError):
    """Exception for tool-related errors."""


class ValidationError(ServiceError):
    """Exception for validation errors."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        input_data: Optional[dict[str, Any]] = None,
        output_data: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the validation error.

        Args:
        ----
            message: Error message
            validation_type: Type of validation that failed
            input_data: Input data that failed validation
            output_data: Output data that failed validation
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.validation_type = validation_type
        self.input_data = input_data
        self.output_data = output_data
        self.with_info(
            validation_type=validation_type,
            input_data=input_data,
            output_data=output_data,
            **kwargs,
        )


class PlanningError(ServiceError):
    """Exception for planning errors."""


class ExecutionError(ServiceError):
    """Exception for execution errors."""

    def __init__(
        self,
        message: str,
        prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the execution error.

        Args:
        ----
            message: Error message
            prompt: The prompt that failed to execute
            model_name: The model that failed to execute the prompt
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.prompt = prompt
        self.model_name = model_name
        self.with_info(prompt=prompt, model_name=model_name, **kwargs)


class OrchestrationError(ServiceError):
    """Exception for orchestration errors."""


class SelfHealingError(ServiceError):
    """Exception for self-healing errors."""


class MonitoringError(ServiceError):
    """Exception for monitoring errors."""


class ModalityError(ServiceError):
    """Exception for modality errors."""


class NetworkError(SaplingsError):
    """Exception for network-related errors."""


class ConnectionError(NetworkError):
    """Exception for connection errors."""


class TimeoutError(NetworkError):
    """Exception for timeout errors."""

    def __init__(
        self,
        message: str,
        elapsed_time: Optional[float] = None,
        operation_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the timeout error.

        Args:
        ----
            message: Error message
            elapsed_time: Time elapsed before timeout occurred
            operation_name: Name of the operation that timed out
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.elapsed_time = elapsed_time
        self.operation_name = operation_name
        self.with_info(elapsed_time=elapsed_time, operation_name=operation_name, **kwargs)


class DataError(SaplingsError):
    """Exception for data-related errors."""


class ParsingError(DataError):
    """Exception for parsing errors."""


class ResourceExhaustedError(SaplingsError):
    """Exception for resource exhaustion (memory, tokens, etc.)."""


class ContextLengthExceededError(ResourceExhaustedError):
    """Exception for context length exceeded errors."""

    def __init__(
        self,
        message: str,
        token_count: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the context length exceeded error.

        Args:
        ----
            message: Error message
            token_count: Actual token count
            max_tokens: Maximum allowed tokens
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.with_info(token_count=token_count, max_tokens=max_tokens, **kwargs)


# Resilience-related exceptions
class ResilienceError(SaplingsError):
    """Base exception for resilience-related errors."""


class OperationTimeoutError(TimeoutError):
    """Exception raised when an operation times out."""


class OperationCancelledError(SaplingsError):
    """Exception raised when an operation is cancelled."""


class CircuitBreakerError(ResilienceError):
    """Exception raised when a circuit breaker is open."""

    def __init__(
        self,
        message: str,
        recovery_time: Optional[float] = None,
        failure_count: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the circuit breaker error.

        Args:
        ----
            message: Error message
            recovery_time: Time until the circuit breaker will try again
            failure_count: Number of failures that triggered the circuit breaker
            **kwargs: Additional error information

        """
        super().__init__(message)
        self.recovery_time = recovery_time
        self.failure_count = failure_count
        self.with_info(recovery_time=recovery_time, failure_count=failure_count, **kwargs)
