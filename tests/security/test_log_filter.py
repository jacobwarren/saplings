from __future__ import annotations

"""
Tests for the log filter module.
"""


import io
import logging
from unittest.mock import patch

from saplings.security.log_filter import RedactingFilter, install_global_filter, redact


def test_redact_api_key() -> None:
    """Test that API keys are properly redacted."""
    sample_text = "Using API key: sk-abcdefghijklmnopqrstuvwx123456789"
    redacted = redact(sample_text)
    assert "sk-abcdefghijklmnopqrstuvwx123456789" not in redacted
    assert "Using API key: ****" in redacted


def test_redact_openai_key() -> None:
    """Test that OpenAI API keys are properly redacted."""
    sample_text = "OPENAI_API_KEY=sk-1234567890abcdefghijklmnopqrstuv"
    redacted = redact(sample_text)
    assert "sk-1234567890abcdefghijklmnopqrstuv" not in redacted
    assert "OPENAI_API_KEY=****" in redacted


def test_redact_aws_key() -> None:
    """Test that AWS keys are properly redacted."""
    sample_text = (
        "Access key: AKIAIOSFODNN7EXAMPLE, Secret: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    redacted = redact(sample_text)
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in redacted
    assert "Access key: ****, Secret: ****" in redacted


def test_redact_github_token() -> None:
    """Test that GitHub tokens are properly redacted."""
    sample_text = "GitHub token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"
    redacted = redact(sample_text)
    assert "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456" not in redacted
    assert "GitHub token: ****" in redacted


def test_redact_slack_token() -> None:
    """Test that Slack tokens are properly redacted."""
    # Using a fake token pattern that won't trigger GitHub's secret scanning
    # Format is modified to avoid detection while still testing the pattern
    sample_text = "Slack bot token: xoxb-FAKE-TEST-TOKEN-aBcDeFgHiJkLmNoPqRsTuVw"
    redacted = redact(sample_text)
    assert "xoxb-FAKE-TEST-TOKEN-aBcDeFgHiJkLmNoPqRsTuVw" not in redacted
    assert "Slack bot token: ****" in redacted


def test_redact_env_var() -> None:
    """Test that environment variables with sensitive names are redacted."""
    sample_text = "DATABASE_PASSWORD=supersecret123"
    redacted = redact(sample_text)
    assert "supersecret123" not in redacted
    assert "DATABASE_PASSWORD=****" in redacted


def test_redact_custom_replacement() -> None:
    """Test using a custom replacement string."""
    sample_text = "API_KEY=abcdefg12345"
    redacted = redact(sample_text, replacement="[REDACTED]")
    assert "API_KEY=[REDACTED]" in redacted


def test_redacting_filter() -> None:
    """Test that the RedactingFilter properly redacts log messages."""
    # Create a log handler that writes to a string buffer
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)

    # Create a logger with our filter
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addFilter(RedactingFilter())

    # Log a message with sensitive information
    logger.info("API key is sk-abcdefghijklmnopqrstuvwxyz1234567890")

    # Get the log output
    log_output = log_buffer.getvalue()

    # Check that the sensitive information is redacted
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in log_output
    assert "API key is ****" in log_output


def test_redacting_filter_args() -> None:
    """Test that the RedactingFilter properly redacts log message arguments."""
    # Create a log handler that writes to a string buffer
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)

    # Create a logger with our filter
    logger = logging.getLogger("test_logger_args")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addFilter(RedactingFilter())

    # Log a message with sensitive information in args
    logger.info("API key is %s", "sk-abcdefghijklmnopqrstuvwxyz1234567890")

    # Get the log output
    log_output = log_buffer.getvalue()

    # Check that the sensitive information is redacted
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in log_output
    assert "API key is ****" in log_output


def test_install_global_filter() -> None:
    """Test installing the filter globally."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_root_logger = mock_get_logger.return_value
        mock_root_logger.filters = []

        # Install the filter
        install_global_filter()

        # Verify it was installed
        mock_root_logger.addFilter.assert_called_once()
        args, _ = mock_root_logger.addFilter.call_args
        assert isinstance(args[0], RedactingFilter)

        # Installing again should not add another filter
        mock_root_logger.filters = [RedactingFilter()]
        install_global_filter()

        # Should still be called only once
        assert mock_root_logger.addFilter.call_count == 1
