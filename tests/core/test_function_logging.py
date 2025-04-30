"""
Tests for function logging.

This module provides tests for the function logging utilities in Saplings.
"""

import logging
import time
from typing import Dict, List, Optional

import pytest

from saplings.core.function_logging import (
    FunctionCallTimer,
    FunctionLogger,
    log_function_call,
    time_function_call,
)


class TestFunctionLogger:
    """Test class for function logging."""

    @pytest.fixture
    def logger(self):
        """Create a function logger for testing."""
        return FunctionLogger(log_level=logging.DEBUG)

    def test_log_function_call(self, logger, caplog):
        """Test logging a function call."""
        # Set up logging
        caplog.set_level(logging.DEBUG)

        # Log a function call
        log_entry = logger.log_function_call(
            name="test_func",
            arguments={"arg1": "value1", "arg2": 2},
            result="test result",
            metadata={"source": "test"}
        )

        # Check the log entry
        assert log_entry["function"] == "test_func"
        assert log_entry["arguments"] == {"arg1": "value1", "arg2": 2}
        assert log_entry["result"] == "test result"
        assert log_entry["status"] == "success"
        assert log_entry["metadata"] == {"source": "test"}
        assert "timestamp" in log_entry
        assert "call_id" in log_entry

        # Check that the log was written
        assert "Function call: test_func" in caplog.text

    def test_log_function_call_with_error(self, logger, caplog):
        """Test logging a function call with an error."""
        # Set up logging
        caplog.set_level(logging.DEBUG)

        # Create an error
        error = ValueError("test error")

        # Log a function call with an error
        log_entry = logger.log_function_call(
            name="test_func",
            arguments={"arg1": "value1"},
            error=error
        )

        # Check the log entry
        assert log_entry["function"] == "test_func"
        assert log_entry["arguments"] == {"arg1": "value1"}
        assert log_entry["status"] == "error"
        assert log_entry["error"] == "test error"
        assert log_entry["error_type"] == "ValueError"

        # Check that the log was written
        assert "Function call: test_func" in caplog.text

    def test_log_function_start_end(self, logger, caplog):
        """Test logging the start and end of a function call."""
        # Set up logging
        caplog.set_level(logging.DEBUG)

        # Log function start
        start_entry = logger.log_function_start(
            name="test_func",
            arguments={"arg1": "value1"},
            metadata={"source": "test"}
        )

        # Check the start entry
        assert start_entry["function"] == "test_func"
        assert start_entry["arguments"] == {"arg1": "value1"}
        assert start_entry["status"] == "started"
        assert start_entry["metadata"] == {"source": "test"}
        assert "timestamp" in start_entry
        assert "call_id" in start_entry

        # Get the call ID
        call_id = start_entry["call_id"]

        # Log function end
        end_entry = logger.log_function_end(
            name="test_func",
            call_id=call_id,
            result="test result",
            duration=0.1
        )

        # Check the end entry
        assert end_entry["function"] == "test_func"
        assert end_entry["call_id"] == call_id
        assert end_entry["status"] == "completed"
        assert end_entry["result"] == "test result"
        assert end_entry["duration"] == 0.1

        # Check that the logs were written
        assert "Function started: test_func" in caplog.text
        assert "Function ended: test_func" in caplog.text

    def test_sanitize_arguments(self, logger):
        """Test sanitizing arguments for logging."""
        # Create arguments with sensitive fields
        arguments = {
            "username": "test_user",
            "password": "secret",
            "api_key": "12345",
            "data": {"secret": "hidden", "public": "visible"}
        }

        # Sanitize the arguments
        sanitized = logger._sanitize_arguments(arguments)

        # Check that sensitive fields are redacted
        assert sanitized["username"] == "test_user"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["data"] == {"secret": "hidden", "public": "visible"}

        # Check that the original arguments are not modified
        assert arguments["password"] == "secret"
        assert arguments["api_key"] == "12345"

    def test_sanitize_result(self, logger):
        """Test sanitizing a result for logging."""
        # Test with simple types
        assert logger._sanitize_result(None) is None
        assert logger._sanitize_result("string") == "string"
        assert logger._sanitize_result(123) == 123
        assert logger._sanitize_result(True) is True

        # Test with a dictionary
        result = {
            "username": "test_user",
            "password": "secret",
            "api_key": "12345"
        }

        sanitized = logger._sanitize_result(result)

        assert sanitized["username"] == "test_user"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"

        # Test with an object
        class TestObject:
            def __str__(self):
                return "test object"

        assert logger._sanitize_result(TestObject()) == "test object"


class TestFunctionCallTimer:
    """Test class for the function call timer."""

    @pytest.fixture
    def logger(self):
        """Create a function logger for testing."""
        return FunctionLogger(log_level=logging.DEBUG)

    def test_function_call_timer(self, logger, caplog):
        """Test timing a function call."""
        # Set up logging
        caplog.set_level(logging.DEBUG)

        # Use the timer as a context manager
        with FunctionCallTimer(
            logger,
            name="test_func",
            arguments={"arg1": "value1"},
            metadata={"source": "test"}
        ):
            # Simulate some work
            time.sleep(0.1)

        # Check that the logs were written
        assert "Function started: test_func" in caplog.text
        assert "Function ended: test_func" in caplog.text

    def test_function_call_timer_with_error(self, logger, caplog):
        """Test timing a function call that raises an error."""
        # Set up logging
        caplog.set_level(logging.DEBUG)

        # Use the timer as a context manager
        try:
            with FunctionCallTimer(
                logger,
                name="test_func",
                arguments={"arg1": "value1"}
            ):
                # Raise an error
                raise ValueError("test error")
        except ValueError:
            pass

        # Check that the logs were written
        assert "Function started: test_func" in caplog.text
        assert "Function ended: test_func" in caplog.text

        # Check for error in the log record
        for record in caplog.records:
            if hasattr(record, 'function_call') and record.function_call.get('status') == 'error':
                assert 'error' in record.function_call
                assert 'test error' in record.function_call['error']
                break
        else:
            assert False, "No error record found in logs"


class TestConvenienceFunctions:
    """Test class for the convenience functions."""

    def test_log_function_call(self, caplog):
        """Test the log_function_call convenience function."""
        # Set up logging
        caplog.set_level(logging.INFO)

        # Log a function call
        log_entry = log_function_call(
            name="test_func",
            arguments={"arg1": "value1"},
            result="test result"
        )

        # Check the log entry
        assert log_entry["function"] == "test_func"
        assert log_entry["arguments"] == {"arg1": "value1"}
        assert log_entry["result"] == "test result"

        # Check that the log was written
        assert "Function call: test_func" in caplog.text

    def test_time_function_call(self, caplog):
        """Test the time_function_call convenience function."""
        # Set up logging
        caplog.set_level(logging.INFO)

        # Use the timer as a context manager
        with time_function_call(
            name="test_func",
            arguments={"arg1": "value1"}
        ):
            # Simulate some work
            time.sleep(0.1)

        # Check that the logs were written
        assert "Function started: test_func" in caplog.text
        assert "Function ended: test_func" in caplog.text
