from __future__ import annotations

"""
Resilience patterns for Saplings.

This module provides standardized resilience patterns for use across the codebase,
including timeout handling, circuit breakers, and retries.
"""

import logging
from typing import Any, Optional, TypeVar

from saplings.core._internal.exceptions import (
    CircuitBreakerError,
    OperationCancelledError,
    OperationTimeoutError,
)
from saplings.core._internal.resilience.resilience import (
    DEFAULT_TIMEOUT,
    CircuitBreaker,
    retry,
    with_timeout,
)
from saplings.core._internal.validation.validation import validate_positive, validate_required

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutPattern:
    """
    Standard timeout pattern for async operations.

    This class provides a standardized way to apply timeouts to async operations,
    with proper error handling and logging.
    """

    @staticmethod
    async def execute(
        coro,
        *,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an async operation with a timeout.

        Args:
        ----
            coro: The coroutine to execute
            timeout: Optional timeout in seconds (None means no timeout)
            operation_name: Name of the operation for logging purposes

        Returns:
        -------
            The result of the coroutine

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled
            Exception: Any other exception raised by the coroutine

        """
        try:
            return await with_timeout(coro, timeout=timeout, operation_name=operation_name)
        except OperationTimeoutError as e:
            logger.warning(f"{operation_name} timed out after {e.elapsed_time:.2f}s")
            raise
        except OperationCancelledError:
            logger.warning(f"{operation_name} was cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in {operation_name}: {e}")
            raise


class RetryPattern:
    """
    Standard retry pattern for async operations.

    This class provides a standardized way to apply retries to async operations,
    with proper error handling, backoff, and logging.
    """

    @staticmethod
    async def execute(
        coro,
        *,
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_exceptions: Optional[list[type[Exception]]] = None,
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an async operation with retries.

        Args:
        ----
            coro: The coroutine to execute
            max_attempts: Maximum number of attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_factor: Factor to multiply backoff by after each attempt
            jitter: Whether to add jitter to backoff times
            retry_exceptions: List of exceptions to retry on (None means all exceptions)
            operation_name: Name of the operation for logging purposes

        Returns:
        -------
            The result of the coroutine

        Raises:
        ------
            Exception: If all retry attempts fail

        """
        # Validate inputs
        validate_positive(max_attempts, "max_attempts")
        validate_positive(initial_backoff, "initial_backoff")
        validate_positive(max_backoff, "max_backoff")
        validate_positive(backoff_factor, "backoff_factor")

        # Create retry decorator
        retry_decorator = retry(
            max_attempts=max_attempts,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            backoff_factor=backoff_factor,
            jitter=jitter,
            retry_exceptions=retry_exceptions,
        )

        # Define a function to execute with retries
        @retry_decorator
        async def execute_with_retries():
            try:
                return await coro
            except Exception as e:
                logger.warning(f"Attempt failed for {operation_name}: {type(e).__name__}: {e}")
                raise

        # Execute with retries
        try:
            return await execute_with_retries()
        except Exception as e:
            logger.exception(f"All retry attempts failed for {operation_name}: {e}")
            raise


class CircuitBreakerPattern:
    """
    Standard circuit breaker pattern for async operations.

    This class provides a standardized way to apply circuit breakers to async operations,
    with proper error handling and logging.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: Optional[list[type[Exception]]] = None,
        name: str = "default",
    ) -> None:
        """
        Initialize the circuit breaker pattern.

        Args:
        ----
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before trying again
            expected_exceptions: List of exceptions that count as failures
            name: Name of the circuit breaker for logging purposes

        """
        self.name = name
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions or [],
        )

    async def execute(self, coro, *, operation_name: str = "operation") -> Any:
        """
        Execute an async operation with circuit breaker protection.

        Args:
        ----
            coro: The coroutine to execute
            operation_name: Name of the operation for logging purposes

        Returns:
        -------
            The result of the coroutine

        Raises:
        ------
            CircuitBreakerError: If the circuit is open
            Exception: Any other exception raised by the coroutine

        """
        try:
            return await self.circuit_breaker.execute(lambda: coro)
        except CircuitBreakerError:
            logger.warning(f"Circuit breaker '{self.name}' is open for {operation_name}")
            raise
        except Exception as e:
            logger.exception(f"Error in {operation_name}: {e}")
            raise


class ResiliencePatterns:
    """
    Facade for all resilience patterns.

    This class provides a single entry point for all resilience patterns,
    making it easy to apply multiple patterns to an operation.
    """

    @staticmethod
    async def with_timeout(
        coro,
        *,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an async operation with a timeout.

        Args:
        ----
            coro: The coroutine to execute
            timeout: Optional timeout in seconds (None means no timeout)
            operation_name: Name of the operation for logging purposes

        Returns:
        -------
            The result of the coroutine

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled
            Exception: Any other exception raised by the coroutine

        """
        return await TimeoutPattern.execute(coro, timeout=timeout, operation_name=operation_name)

    @staticmethod
    async def with_retry(
        coro,
        *,
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_exceptions: Optional[list[type[Exception]]] = None,
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an async operation with retries.

        Args:
        ----
            coro: The coroutine to execute
            max_attempts: Maximum number of attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_factor: Factor to multiply backoff by after each attempt
            jitter: Whether to add jitter to backoff times
            retry_exceptions: List of exceptions to retry on (None means all exceptions)
            operation_name: Name of the operation for logging purposes

        Returns:
        -------
            The result of the coroutine

        Raises:
        ------
            Exception: If all retry attempts fail

        """
        return await RetryPattern.execute(
            coro,
            max_attempts=max_attempts,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            backoff_factor=backoff_factor,
            jitter=jitter,
            retry_exceptions=retry_exceptions,
            operation_name=operation_name,
        )

    @staticmethod
    async def with_circuit_breaker(
        coro,
        *,
        circuit_breaker: CircuitBreakerPattern,
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an async operation with circuit breaker protection.

        Args:
        ----
            coro: The coroutine to execute
            circuit_breaker: The circuit breaker to use
            operation_name: Name of the operation for logging purposes

        Returns:
        -------
            The result of the coroutine

        Raises:
        ------
            CircuitBreakerError: If the circuit is open
            Exception: Any other exception raised by the coroutine

        """
        validate_required(circuit_breaker, "circuit_breaker")
        return await circuit_breaker.execute(coro, operation_name=operation_name)

    @staticmethod
    async def with_all(
        coro,
        *,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        max_attempts: int = 3,
        circuit_breaker: Optional[CircuitBreakerPattern] = None,
        operation_name: str = "operation",
    ) -> Any:
        """
        Execute an async operation with all resilience patterns.

        Args:
        ----
            coro: The coroutine to execute
            timeout: Optional timeout in seconds (None means no timeout)
            max_attempts: Maximum number of retry attempts
            circuit_breaker: Optional circuit breaker to use
            operation_name: Name of the operation for logging purposes

        Returns:
        -------
            The result of the coroutine

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            CircuitBreakerError: If the circuit is open
            Exception: If all retry attempts fail

        """
        # Apply circuit breaker if provided
        if circuit_breaker is not None:
            coro = circuit_breaker.execute(coro, operation_name=operation_name)

        # Apply retry
        coro = RetryPattern.execute(coro, max_attempts=max_attempts, operation_name=operation_name)

        # Apply timeout
        return await TimeoutPattern.execute(coro, timeout=timeout, operation_name=operation_name)
