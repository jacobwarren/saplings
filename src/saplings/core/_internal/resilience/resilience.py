from __future__ import annotations

"""
Resilience module for Saplings.

This module provides utilities for handling sync/async boundaries, timeouts,
and cancellation in a consistent way across the codebase.
"""


import asyncio
import concurrent.futures
import functools
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from saplings.core._internal.exceptions import (
    CircuitBreakerError,
    OperationCancelledError,
    OperationTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)

# Type variables for generic type hints
T = TypeVar("T")
R = TypeVar("R")

# Default executor for running blocking functions
_thread_pool = concurrent.futures.ThreadPoolExecutor()

# Global configuration
DEFAULT_TIMEOUT = 60.0  # Default timeout in seconds - increased to avoid cancellation issues


class Validation:
    """
    Utility class for validating inputs.

    This class provides static methods for common validation operations.
    """

    @staticmethod
    def require(condition: bool, message: str) -> None:
        """
        Require that a condition is true.

        Args:
        ----
            condition: Condition to check
            message: Error message if condition is false

        Raises:
        ------
            ValueError: If the condition is false

        """
        if not condition:
            raise ValueError(message)

    @staticmethod
    def require_not_empty(value: Any, name: str) -> None:
        """
        Require that a value is not empty.

        Args:
        ----
            value: Value to check
            name: Name of the value for error message

        Raises:
        ------
            ValueError: If the value is empty

        """
        if value is None:
            msg = f"{name} cannot be None"
            raise ValueError(msg)

        if isinstance(value, str) and not value.strip():
            msg = f"{name} cannot be empty"
            raise ValueError(msg)

        if hasattr(value, "__len__") and len(value) == 0:
            msg = f"{name} cannot be empty"
            raise ValueError(msg)

    @staticmethod
    def require_not_none(value: Any, name: str) -> None:
        """
        Require that a value is not None.

        Args:
        ----
            value: Value to check
            name: Name of the value for error message

        Raises:
        ------
            ValueError: If the value is None

        """
        if value is None:
            msg = f"{name} cannot be None"
            raise ValueError(msg)

    @staticmethod
    def require_positive(value: float, name: str) -> None:
        """
        Require that a value is positive.

        Args:
        ----
            value: Value to check
            name: Name of the value for error message

        Raises:
        ------
            ValueError: If the value is not positive

        """
        if value <= 0:
            msg = f"{name} must be positive"
            raise ValueError(msg)

    @staticmethod
    def require_non_negative(value: float, name: str) -> None:
        """
        Require that a value is non-negative.

        Args:
        ----
            value: Value to check
            name: Name of the value for error message

        Raises:
        ------
            ValueError: If the value is negative

        """
        # Threshold for value
        value_threshold = 0
        if value < value_threshold:
            msg = f"{name} cannot be negative"
            raise ValueError(msg)


# Exception classes are now imported from saplings.core.exceptions


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    This class implements the circuit breaker pattern to prevent
    repeated calls to a failing service, allowing it time to recover.
    """

    # Circuit states
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if the service has recovered

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: list[type[Exception]] | None = None,
    ) -> None:
        """
        Initialize the circuit breaker.

        Args:
        ----
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before trying again
            expected_exceptions: List of exceptions that count as failures

        """
        # Validate inputs
        Validation.require_positive(failure_threshold, "failure_threshold")
        Validation.require_positive(recovery_timeout, "recovery_timeout")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions or []

        # State
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
        ----
            func: Async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
        -------
            The result of the function call

        Raises:
        ------
            CircuitBreakerError: If the circuit is open
            Exception: Any exception raised by the function

        """
        async with self._lock:
            # Check if the circuit is open
            if self.state == self.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    # Transition to half-open state
                    self.state = self.HALF_OPEN
                    logger.info("Circuit breaker transitioning to half-open state")
                else:
                    # Circuit is still open, fail fast
                    msg = (
                        f"Circuit breaker is open. Failing fast. "
                        f"Will try again in {self.recovery_timeout - (time.time() - self.last_failure_time):.2f}s"
                    )
                    recovery_time = self.recovery_timeout - (time.time() - self.last_failure_time)
                    raise CircuitBreakerError(
                        msg, recovery_time=recovery_time, failure_count=self.failure_count
                    )

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Record success
            async with self._lock:
                self.last_success_time = time.time()

                # If we were in half-open state, transition to closed
                if self.state == self.HALF_OPEN:
                    self.state = self.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker closed after successful recovery")

            return result

        except Exception as e:
            # Check if this is an expected exception
            is_expected = any(isinstance(e, exc_type) for exc_type in self.expected_exceptions)

            if not is_expected:
                # Not an expected exception, re-raise without affecting the circuit
                raise

            # Record failure
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                # Check if we need to open the circuit
                if self.state == self.CLOSED and self.failure_count >= self.failure_threshold:
                    self.state = self.OPEN
                    logger.warning(
                        f"Circuit breaker opened after {self.failure_count} failures. "
                        f"Will try again in {self.recovery_timeout}s"
                    )

                # If we were in half-open state, go back to open
                elif self.state == self.HALF_OPEN:
                    self.state = self.OPEN
                    logger.warning(
                        f"Circuit breaker reopened after failed recovery attempt. "
                        f"Will try again in {self.recovery_timeout}s"
                    )

            # Re-raise the exception
            raise

    def reset(self):
        """Reset the circuit breaker to closed state."""
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        logger.info("Circuit breaker reset to closed state")


# OperationCancelledError is now imported from saplings.core.exceptions


async def run_in_executor(
    func: Callable[..., R],
    *args,
    executor: concurrent.futures.Executor | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    **kwargs,
) -> R:
    """
    Run a blocking function in an executor with timeout support.

    Args:
    ----
        func: The blocking function to run
        *args: Arguments to pass to the function
        executor: Optional executor to use (defaults to thread pool)
        timeout: Optional timeout in seconds (None means no timeout)
        **kwargs: Keyword arguments to pass to the function

    Returns:
    -------
        The result of the function call

    Raises:
    ------
        OperationTimeoutError: If the operation times out
        OperationCancelledError: If the operation is cancelled
        Exception: Any exception raised by the function

    """
    start_time = time.time()

    # Use the provided executor or the default thread pool
    executor = executor or _thread_pool

    try:
        # Run the blocking function in the executor
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                executor, functools.partial(func, *args, **kwargs)
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(f"Operation timed out after {elapsed:.2f}s: {func.__name__}")
        msg = f"Operation timed out after {elapsed:.2f}s: {func.__name__}"
        raise OperationTimeoutError(msg, elapsed_time=elapsed, operation_name=func.__name__)
    except asyncio.CancelledError:
        logger.warning(f"Operation cancelled: {func.__name__}")
        msg = f"Operation cancelled: {func.__name__}"
        raise OperationCancelledError(msg)


async def with_timeout(
    awaitable: Awaitable[T],
    timeout: float | None = DEFAULT_TIMEOUT,
    operation_name: str = "operation",
) -> T:
    """
    Execute an awaitable with a timeout.

    This function properly handles timeouts and cancellations, ensuring that
    resources are cleaned up properly and that errors are propagated correctly.

    Args:
    ----
        awaitable: The awaitable to execute
        timeout: Optional timeout in seconds (None means no timeout)
        operation_name: Name of the operation for logging purposes

    Returns:
    -------
        The result of the awaitable

    Raises:
    ------
        OperationTimeoutError: If the operation times out
        OperationCancelledError: If the operation is cancelled
        Exception: Any exception raised by the awaitable

    """
    if timeout is None:
        # If no timeout is specified, just await the awaitable directly
        return await awaitable

    start_time = time.time()

    # Create a task for the awaitable to ensure we can cancel it properly
    # Handle both coroutines and other awaitables
    if asyncio.iscoroutine(awaitable):
        task = asyncio.create_task(awaitable)
    else:
        # For other awaitables, wrap in a coroutine
        async def _wrap_awaitable():
            return await awaitable

        task = asyncio.create_task(_wrap_awaitable())

    try:
        # Use asyncio.wait_for with the task
        # This allows us to properly cancel the task if a timeout occurs
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        # Handle timeout
        elapsed = time.time() - start_time
        logger.warning(f"{operation_name} timed out after {elapsed:.2f}s")

        # Cancel the task if it's still running
        if not task.done():
            task.cancel()

            # Wait for the task to be cancelled (with a short timeout)
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                # Ignore any errors during cancellation
                pass

        msg = f"{operation_name} timed out after {elapsed:.2f}s"
        raise OperationTimeoutError(msg, elapsed_time=elapsed, operation_name=operation_name)
    except asyncio.CancelledError:
        # Handle cancellation
        logger.warning(f"{operation_name} cancelled")

        # Cancel the task if it's still running
        if not task.done():
            task.cancel()

            # Wait for the task to be cancelled (with a short timeout)
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                # Ignore any errors during cancellation
                pass

        msg = f"{operation_name} cancelled"
        raise OperationCancelledError(msg)
    except Exception as e:
        # Handle other exceptions
        logger.exception(f"Error in {operation_name}: {e}")
        raise


@asynccontextmanager
async def timeout_context(
    timeout: float | None = DEFAULT_TIMEOUT, operation_name: str = "operation"
):
    """
    Context manager for operations that need timeout handling.

    Args:
    ----
        timeout: Optional timeout in seconds (None means no timeout)
        operation_name: Name of the operation for logging purposes

    Yields:
    ------
        None

    Raises:
    ------
        OperationTimeoutError: If the operation times out
        OperationCancelledError: If the operation is cancelled

    """
    start_time = time.time()

    if timeout is None:
        yield
        return

    try:
        # Create a task for the timeout
        task = asyncio.create_task(asyncio.sleep(timeout))

        try:
            yield
            # If we get here without timing out, cancel the timeout task
            task.cancel()
        except asyncio.CancelledError:
            logger.warning(f"{operation_name} cancelled")
            msg = f"{operation_name} cancelled"
            raise OperationCancelledError(msg)
    finally:
        if not task.done():
            task.cancel()
            try:
                # Use asyncio.wait_for to await the task
                await asyncio.wait_for(task, timeout=None)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    # Check if we timed out
    if task.done() and not task.cancelled():
        elapsed = time.time() - start_time
        logger.warning(f"{operation_name} timed out after {elapsed:.2f}s")
        msg = f"{operation_name} timed out after {elapsed:.2f}s"
        raise OperationTimeoutError(msg, elapsed_time=elapsed, operation_name=operation_name)


def configure_executor(max_workers: int | None = None) -> None:
    """
    Configure the global thread pool executor.

    Args:
    ----
        max_workers: Maximum number of worker threads

    """
    global _thread_pool
    _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


def retry(
    max_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: list[type[Exception]] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
    ----
        max_attempts: Maximum number of attempts
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_factor: Factor to multiply backoff by after each attempt
        jitter: Whether to add jitter to backoff times
        retry_exceptions: List of exceptions to retry on (None means all exceptions)

    Returns:
    -------
        Decorator function

    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            backoff = initial_backoff

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if retry_exceptions is not None and not any(
                        isinstance(e, exc_type) for exc_type in retry_exceptions
                    ):
                        # Not a retryable exception
                        raise

                    # Last attempt, re-raise the exception
                    if attempt >= max_attempts - 1:
                        raise

                    # Calculate backoff with optional jitter
                    if jitter:
                        # Add up to 20% jitter
                        jitter_amount = (
                            backoff * 0.2 * (2 * asyncio.get_event_loop().time() % 1 - 1)
                        )
                        sleep_time = min(backoff + jitter_amount, max_backoff)
                    else:
                        sleep_time = min(backoff, max_backoff)

                    # Log the retry
                    logger.warning(
                        f"Retrying {func.__name__} after error: {e}. "
                        f"Attempt {attempt + 1}/{max_attempts}. "
                        f"Waiting {sleep_time:.2f}s before next attempt."
                    )

                    # Wait before retrying
                    await asyncio.sleep(sleep_time)

                    # Increase backoff for next attempt
                    backoff = min(backoff * backoff_factor, max_backoff)

            # This should never happen, but just in case
            if last_exception:
                raise last_exception
            msg = "Unexpected error in retry logic"
            raise RuntimeError(msg)

        return wrapper

    return decorator


def with_timeout_decorator(timeout: float | None = DEFAULT_TIMEOUT):
    """
    Decorator that applies a timeout to an async function.

    Args:
    ----
        timeout: Optional timeout in seconds (None means no timeout)

    Returns:
    -------
        Decorator function

    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await with_timeout(
                func(*args, **kwargs), timeout=timeout, operation_name=func.__name__
            )

        return wrapper

    return decorator


def get_executor():
    """
    Get the global thread pool executor.

    Returns
    -------
        The global thread pool executor

    """
    return _thread_pool
