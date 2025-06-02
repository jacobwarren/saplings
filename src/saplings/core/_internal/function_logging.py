from __future__ import annotations

"""
Function logging module for Saplings.

This module provides utilities for logging function calls.
"""


import logging
import time
import uuid
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class FunctionLogger:
    """Utility for logging function calls."""

    def __init__(self, log_level: int = logging.INFO) -> None:
        """
        Initialize the function logger.

        Args:
        ----
            log_level: Logging level

        """
        self.log_level = log_level
        self._function_logger = logging.getLogger("saplings.function_calls")
        self._function_logger.setLevel(log_level)

    def log_function_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
        duration: float | None = None,
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Log a function call.

        Args:
        ----
            name: Name of the function
            arguments: Arguments passed to the function
            result: Result of the function call
            error: Error raised by the function
            duration: Duration of the function call in seconds
            call_id: Unique ID for the function call
            metadata: Additional metadata

        Returns:
        -------
            Dict[str, Any]: Log entry

        """
        # Create a log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id or str(uuid.uuid4()),
            "function": name,
            "arguments": self._sanitize_arguments(arguments),
            "metadata": metadata or {},
        }

        # Add result or error
        if error:
            log_entry["status"] = "error"
            log_entry["error"] = str(error)
            log_entry["error_type"] = error.__class__.__name__
        else:
            log_entry["status"] = "success"
            log_entry["result"] = self._sanitize_result(result)

        # Add duration if provided
        if duration is not None:
            log_entry["duration"] = duration

        # Log the entry
        self._function_logger.log(
            self.log_level, f"Function call: {name}", extra={"function_call": log_entry}
        )

        return log_entry

    def log_function_start(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Log the start of a function call.

        Args:
        ----
            name: Name of the function
            arguments: Arguments passed to the function
            call_id: Unique ID for the function call
            metadata: Additional metadata

        Returns:
        -------
            Dict[str, Any]: Log entry

        """
        # Create a log entry
        call_id = call_id or str(uuid.uuid4())
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "function": name,
            "arguments": self._sanitize_arguments(arguments),
            "metadata": metadata or {},
            "status": "started",
        }

        # Log the entry
        self._function_logger.log(
            self.log_level, f"Function started: {name}", extra={"function_call": log_entry}
        )

        return log_entry

    def log_function_end(
        self,
        name: str,
        call_id: str,
        result: Any = None,
        error: Exception | None = None,
        duration: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Log the end of a function call.

        Args:
        ----
            name: Name of the function
            call_id: Unique ID for the function call
            result: Result of the function call
            error: Error raised by the function
            duration: Duration of the function call in seconds
            metadata: Additional metadata

        Returns:
        -------
            Dict[str, Any]: Log entry

        """
        # Create a log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "function": name,
            "metadata": metadata or {},
        }

        # Add result or error
        if error:
            log_entry["status"] = "error"
            log_entry["error"] = str(error)
            log_entry["error_type"] = error.__class__.__name__
        else:
            log_entry["status"] = "completed"
            log_entry["result"] = self._sanitize_result(result)

        # Add duration if provided
        if duration is not None:
            log_entry["duration"] = duration

        # Log the entry
        self._function_logger.log(
            self.log_level, f"Function ended: {name}", extra={"function_call": log_entry}
        )

        return log_entry

    def _sanitize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize arguments for logging.

        Args:
        ----
            arguments: Arguments to sanitize

        Returns:
        -------
            Dict[str, Any]: Sanitized arguments

        """
        # Make a copy to avoid modifying the original
        sanitized = arguments.copy()

        # Sanitize sensitive fields
        sensitive_fields = ["password", "api_key", "secret", "token", "auth"]
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***REDACTED***"

        return sanitized

    def _sanitize_result(self, result: Any) -> Any:
        """
        Sanitize a result for logging.

        Args:
        ----
            result: Result to sanitize

        Returns:
        -------
            Any: Sanitized result

        """
        # For simple types, return as is
        if result is None or isinstance(result, (str, int, float, bool)):
            return result

        # For dictionaries, sanitize sensitive fields
        if isinstance(result, dict):
            sanitized = result.copy()
            sensitive_fields = ["password", "api_key", "secret", "token", "auth"]
            for field in sensitive_fields:
                if field in sanitized:
                    sanitized[field] = "***REDACTED***"
            return sanitized

        # For other types, convert to string
        try:
            return str(result)
        except Exception:
            return "***UNLOGGABLE***"


class FunctionCallTimer:
    """Context manager for timing function calls."""

    def __init__(
        self,
        function_logger: FunctionLogger,
        name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the function call timer.

        Args:
        ----
            function_logger: Function logger
            name: Name of the function
            arguments: Arguments passed to the function
            metadata: Additional metadata

        """
        self.function_logger = function_logger
        self.name = name
        self.arguments = arguments
        self.metadata = metadata or {}
        self.start_time = None
        self.call_id = None
        self.log_entry = None

    def __enter__(self):
        """Start timing the function call."""
        self.start_time = time.time()
        self.log_entry = self.function_logger.log_function_start(
            self.name, self.arguments, metadata=self.metadata
        )
        self.call_id = self.log_entry["call_id"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing the function call."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time is not None else None

        call_id = self.call_id if self.call_id is not None else ""
        self.function_logger.log_function_end(
            self.name,
            call_id,
            result=None if exc_val else "***RESULT_NOT_CAPTURED***",
            error=exc_val,
            duration=duration,
            metadata=self.metadata,
        )


# Create a singleton instance
function_logger = FunctionLogger()


def log_function_call(
    name: str,
    arguments: dict[str, Any],
    result: Any = None,
    error: Exception | None = None,
    duration: float | None = None,
    call_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Log a function call.

    This is a convenience function that uses the FunctionLogger.

    Args:
    ----
        name: Name of the function
        arguments: Arguments passed to the function
        result: Result of the function call
        error: Error raised by the function
        duration: Duration of the function call in seconds
        call_id: Unique ID for the function call
        metadata: Additional metadata

    Returns:
    -------
        Dict[str, Any]: Log entry

    """
    return function_logger.log_function_call(
        name, arguments, result, error, duration, call_id, metadata
    )


def time_function_call(
    name: str, arguments: dict[str, Any], metadata: dict[str, Any] | None = None
) -> FunctionCallTimer:
    """
    Time a function call.

    This is a convenience function that returns a context manager.

    Args:
    ----
        name: Name of the function
        arguments: Arguments passed to the function
        metadata: Additional metadata

    Returns:
    -------
        FunctionCallTimer: Context manager for timing the function call

    """
    return FunctionCallTimer(function_logger, name, arguments, metadata)
