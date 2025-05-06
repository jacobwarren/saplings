from __future__ import annotations

"""
Tests for API key protection.

These tests verify that API keys are properly protected in logs and error messages.
"""


import io
import logging

from saplings.security.log_filter import RedactingFilter, install_global_filter, redact


class TestAPIKeyProtection:
    """Test API key protection."""

    def test_redact_openai_api_key(self) -> None:
        """Test that OpenAI API keys are properly redacted."""
        # Test with OpenAI API key
        sample_text = "Using OpenAI API key: sk-abcdefghijklmnopqrstuvwx123456789"
        redacted = redact(sample_text)
        assert "sk-abcdefghijklmnopqrstuvwx123456789" not in redacted
        assert "Using OpenAI API key: ****" in redacted

    def test_redact_anthropic_api_key(self) -> None:
        """Test that Anthropic API keys are properly redacted."""
        # Test with Anthropic API key
        sample_text = "Using Anthropic API key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        redacted = redact(sample_text)
        assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "Using Anthropic API key: ****" in redacted

    def test_redact_huggingface_api_key(self) -> None:
        """Test that HuggingFace API keys are properly redacted."""
        # Test with HuggingFace API key
        sample_text = "Using HuggingFace API key: hf_abcdefghijklmnopqrstuvwxyz"
        redacted = redact(sample_text)
        assert "hf_abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "Using HuggingFace API key: ****" in redacted

    def test_redact_multiple_api_keys(self) -> None:
        """Test that multiple API keys are properly redacted."""
        # Test with multiple API keys
        sample_text = """
        OpenAI API key: sk-abcdefghijklmnopqrstuvwx123456789
        Anthropic API key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz
        HuggingFace API key: hf_abcdefghijklmnopqrstuvwxyz
        """
        redacted = redact(sample_text)
        assert "sk-abcdefghijklmnopqrstuvwx123456789" not in redacted
        assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "hf_abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "OpenAI API key: ****" in redacted
        assert "Anthropic API key: ****" in redacted
        assert "HuggingFace API key: ****" in redacted

    def test_redacting_filter(self) -> None:
        """Test the RedactingFilter for log messages."""
        # Create a logger
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)

        # Create a string IO for capturing log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        # Add the redacting filter
        handler.addFilter(RedactingFilter())

        # Add the handler to the logger
        logger.addHandler(handler)

        # Log a message with an API key
        logger.info("Using OpenAI API key: sk-abcdefghijklmnopqrstuvwx123456789")

        # Get the log output
        log_output = log_capture.getvalue()

        # Verify the API key is redacted
        assert "sk-abcdefghijklmnopqrstuvwx123456789" not in log_output
        assert "Using OpenAI API key: ****" in log_output

    def test_install_global_filter(self) -> None:
        """Test installing the global filter."""
        # Install the global filter
        install_global_filter()

        # Create a string IO for capturing log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        # Get the root logger and add the handler
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        # Log a message with an API key
        logging.info("Using OpenAI API key: sk-abcdefghijklmnopqrstuvwx123456789")

        # Get the log output
        log_output = log_capture.getvalue()

        # Verify the API key is redacted
        assert "sk-abcdefghijklmnopqrstuvwx123456789" not in log_output
        assert "Using OpenAI API key: ****" in log_output

        # Clean up
        root_logger.removeHandler(handler)
