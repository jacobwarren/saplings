from __future__ import annotations

"""
Log filtering module for Saplings.

This module provides utilities for filtering sensitive information from logs.
"""


import logging
import re
from re import Pattern
from typing import Any, Dict, List, Optional


class RedactingFilter(logging.Filter):
    """
    A logging filter that redacts sensitive information.

    This filter can be added to any logger to automatically redact sensitive
    information like API keys, passwords, and other credentials from log messages.
    """

    # Default patterns to redact
    DEFAULT_PATTERNS = [
        # API keys and tokens
        r"(api[_-]?key|api[_-]?token|access[_-]?token|auth[_-]?token)[\"\']?\s*[=:]\s*[\"\']?([^\"\'\s]+)",
        # Passwords
        r"(password|passwd|pwd)[\"\']?\s*[=:]\s*[\"\']?([^\"\'\s]+)",
        # Credit card numbers (basic pattern)
        r"\b(?:\d{4}[- ]?){3}\d{4}\b",
        # Social security numbers
        r"\b\d{3}-\d{2}-\d{4}\b",
        # Email addresses
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # Bearer tokens
        r"Bearer\s+([A-Za-z0-9-._~+/]+=*)",
        # Basic auth
        r"Basic\s+([A-Za-z0-9+/]+=*)",
        # OpenAI API keys
        r"sk-[A-Za-z0-9]{48}",
        # Anthropic API keys
        r"sk-ant-[A-Za-z0-9]{48}",
        # AWS access keys
        r"AKIA[0-9A-Z]{16}",
        # Generic secrets
        r"secret[\"\']?\s*[=:]\s*[\"\']?([^\"\'\s]+)",
    ]

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        replacement: str = "[REDACTED]",
    ) -> None:
        """
        Initialize the redacting filter.

        Args:
        ----
            patterns: List of regex patterns to redact. If None, uses default patterns.
            replacement: String to replace sensitive information with.

        """
        self.replacement = replacement
        self.patterns: List[Pattern[str]] = []

        # Compile patterns
        for pattern in patterns or self.DEFAULT_PATTERNS:
            try:
                self.patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                # Skip invalid patterns
                pass

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records by redacting sensitive information.

        Args:
        ----
            record: Log record to filter

        Returns:
        -------
            bool: Always True (to keep the record)

        """
        if isinstance(record.msg, str):
            # Redact the message
            record.msg = self._redact(record.msg)

        # Redact args if they are strings
        if record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    args[i] = self._redact(arg)
                elif isinstance(arg, dict):
                    # Redact dictionary values
                    args[i] = self._redact_dict(arg)
            record.args = tuple(args)

        return True

    def _redact(self, text: str) -> str:
        """
        Redact sensitive information from text.

        Args:
        ----
            text: Text to redact

        Returns:
        -------
            str: Redacted text

        """
        for pattern in self.patterns:
            text = pattern.sub(f"\\1={self.replacement}", text)
        return text

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from a dictionary.

        Args:
        ----
            data: Dictionary to redact

        Returns:
        -------
            Dict[str, Any]: Redacted dictionary

        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._redact(value)
            elif isinstance(value, dict):
                result[key] = self._redact_dict(value)
            else:
                result[key] = value
        return result


def redact(text: str, patterns: Optional[List[str]] = None, replacement: str = "[REDACTED]") -> str:
    """
    Redact sensitive information from text.

    Args:
    ----
        text: Text to redact
        patterns: List of regex patterns to redact. If None, uses default patterns.
        replacement: String to replace sensitive information with.

    Returns:
    -------
        str: Redacted text

    """
    filter_obj = RedactingFilter(patterns=patterns, replacement=replacement)
    return filter_obj._redact(text)


def install_global_filter(
    patterns: Optional[List[str]] = None, replacement: str = "[REDACTED]"
) -> None:
    """
    Install a global redacting filter on the root logger.

    Args:
    ----
        patterns: List of regex patterns to redact. If None, uses default patterns.
        replacement: String to replace sensitive information with.

    """
    root_logger = logging.getLogger()
    filter_obj = RedactingFilter(patterns=patterns, replacement=replacement)
    root_logger.addFilter(filter_obj)
