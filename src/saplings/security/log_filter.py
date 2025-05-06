from __future__ import annotations

"""
Log filtering utilities for Saplings.

This module provides utilities to filter sensitive information from logs,
such as API keys, tokens, and other credentials.
"""


import logging
import re
from re import Pattern

# Regular expressions for identifying potential secrets
# This covers common patterns for API keys, tokens, and credentials
_KEY_PATTERN = re.compile(
    r"(?:sk|pk|key|token|secret|password|credential|auth)[A-Za-z0-9_\-]{10,}", re.IGNORECASE
)

# Pattern for common API key formats (OpenAI, AWS, etc.)
_API_KEY_PATTERN = re.compile(r"(?:sk-|AKIA|ghp_|xoxp-|xoxb-|xapp-)[A-Za-z0-9_\-]{10,}")

# Environment variable names that often contain secrets
_ENV_VAR_PATTERN = re.compile(r"([A-Z_][A-Z0-9_]*)=(.*)")


def redact(text: str, replacement: str = "****") -> str:
    """
    Redact sensitive information in text.

    Args:
    ----
        text: The text to redact
        replacement: The string to use as a replacement for sensitive information

    Returns:
    -------
        Redacted text

    """
    if not text:
        return ""

    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)

    # Redact API keys and tokens
    redacted = _KEY_PATTERN.sub(replacement, text)
    redacted = _API_KEY_PATTERN.sub(replacement, redacted)

    # Redact environment variables containing sensitive data
    def _env_var_replacer(match):
        var_name = match.group(1)
        var_value = match.group(2)

        # Check if the variable name might indicate sensitive data
        sensitive_prefixes = ["API", "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"]
        for prefix in sensitive_prefixes:
            if prefix in var_name:
                return f"{var_name}={replacement}"

        # Also check the value format for typical API key patterns
        if _KEY_PATTERN.search(var_value) or _API_KEY_PATTERN.search(var_value):
            return f"{var_name}={replacement}"

        return match.group(0)

    return _ENV_VAR_PATTERN.sub(_env_var_replacer, redacted)


class RedactingFilter(logging.Filter):
    """
    Logging filter that redacts sensitive information.

    This filter can be attached to any logger to ensure sensitive information
    like API keys, tokens, and credentials are not logged.
    """

    def __init__(
        self,
        name: str = "",
        patterns: list[Pattern[str]] | None = None,
        replacement: str = "****",
    ) -> None:
        """
        Initialize the filter.

        Args:
        ----
            name: Name of the filter
            patterns: Additional patterns to redact beyond the defaults
            replacement: String to use as replacement for redacted content

        """
        super().__init__(name)
        self.patterns = patterns or []
        self.replacement = replacement

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter a log record.

        This method redacts sensitive information in the log message and arguments.

        Args:
        ----
            record: The log record to filter

        Returns:
        -------
            True to allow the record to be emitted, False to block it

        """
        # Redact the message
        if isinstance(record.msg, str):
            record.msg = redact(record.msg, self.replacement)

        # Redact the arguments
        if record.args:
            args = []
            for arg in record.args:
                if isinstance(arg, str):
                    args.append(redact(arg, self.replacement))
                elif isinstance(arg, (dict, list, tuple)):
                    # For collections, we need to convert to string, redact, then back
                    args.append(redact(str(arg), self.replacement))
                else:
                    args.append(arg)

            record.args = tuple(args)

        # Redact exception info if present
        if record.exc_info and record.exc_text:
            record.exc_text = redact(record.exc_text, self.replacement)

        return True


def install_global_filter(
    patterns: list[Pattern[str]] | None = None, replacement: str = "****"
) -> None:
    """
    Install the redacting filter globally on the root logger.

    Args:
    ----
        patterns: Additional patterns to redact beyond the defaults
        replacement: String to use as replacement for redacted content

    """
    root_logger = logging.getLogger()

    # Check if filter is already installed
    for filter in root_logger.filters:
        if isinstance(filter, RedactingFilter):
            # Already installed
            return

    # Add the filter
    filter = RedactingFilter(patterns=patterns, replacement=replacement)
    root_logger.addFilter(filter)

    logging.info("Installed global log redaction filter")
