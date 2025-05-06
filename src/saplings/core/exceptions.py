from __future__ import annotations

"""
Exception classes for Saplings.

This module provides custom exception classes for Saplings to enable
more granular error handling and preserve stack traces.
"""


from typing import Any


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


class ModelError(SaplingsError):
    """Base exception for model-related errors."""


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


class MissingDependencyError(SaplingsError):
    """Exception for missing dependencies."""


class MemoryError(ServiceError):
    """Exception for memory management errors."""


class IndexError(MemoryError):
    """Exception for indexing errors."""


class RetrievalError(ServiceError):
    """Exception for retrieval errors."""


class ToolError(ServiceError):
    """Exception for tool-related errors."""


class ValidationError(ServiceError):
    """Exception for validation errors."""


class PlanningError(ServiceError):
    """Exception for planning errors."""


class ExecutionError(ServiceError):
    """Exception for execution errors."""


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
