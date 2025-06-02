from __future__ import annotations

"""
Public API for security.

This module provides the public API for security utilities, including:
- Sanitization to prevent injection attacks
- Log filtering to redact sensitive information
- Security validation utilities
- Import hooks for detecting usage of internal APIs
"""

from typing import Any

from saplings.api.stability import beta
from saplings.security._internal.hooks.import_hook import (
    install_import_hook as _install_import_hook,
)
from saplings.security._internal.logging.log_filter import (
    RedactingFilter as _RedactingFilter,
)
from saplings.security._internal.logging.log_filter import (
    install_global_filter as _install_global_filter,
)
from saplings.security._internal.logging.log_filter import (
    redact as _redact,
)
from saplings.security._internal.sanitization.sanitizer import (
    Sanitizer as _Sanitizer,
)
from saplings.security._internal.sanitization.sanitizer import (
    sanitize as _sanitize,
)


@beta
class RedactingFilter(_RedactingFilter):
    """
    Logging filter that redacts sensitive information.

    This filter can be attached to any logger to ensure sensitive information
    like API keys, tokens, and credentials are not logged.
    """


@beta
class Sanitizer(_Sanitizer):
    """
    Sanitizer for preventing prompt injection attacks.

    This class provides methods for sanitizing inputs to prevent
    prompt injection attacks and other security issues.
    """


@beta
def redact(text: str, patterns: list[str] | None = None, replacement: str = "****") -> str:
    """
    Redact sensitive information in text.

    Args:
    ----
        text: The text to redact
        patterns: List of regex patterns to redact. If None, uses default patterns.
        replacement: The string to use as a replacement for sensitive information

    Returns:
    -------
        Redacted text

    """
    return _redact(text, patterns=patterns, replacement=replacement)


@beta
def sanitize(
    input_data: str | dict[str, Any] | list[Any],
    injection_patterns: list[str] | None = None,
    allowed_tags: set[str] | None = None,
    max_input_length: int = 32768,
) -> str | dict[str, Any] | list[Any]:
    """
    Sanitize input data.

    Args:
    ----
        input_data: Data to sanitize
        injection_patterns: List of regex patterns to detect injections
        allowed_tags: Set of HTML/XML tags that are allowed
        max_input_length: Maximum allowed input length

    Returns:
    -------
        Sanitized data

    """
    return _sanitize(
        input_data,
        injection_patterns=injection_patterns,
        allowed_tags=allowed_tags,
        max_input_length=max_input_length,
    )


@beta
def install_global_filter(**kwargs) -> None:
    """
    Install the redacting filter globally on the root logger.

    Args:
    ----
        **kwargs: Additional arguments to pass to the filter

    """
    return _install_global_filter(**kwargs)


@beta
def install_import_hook() -> None:
    """
    Install the import hook for detecting usage of internal APIs.

    This function installs an import hook that detects imports of internal
    modules and symbols, and issues deprecation warnings when they are used.
    """
    return _install_import_hook()


__all__ = [
    "RedactingFilter",
    "Sanitizer",
    "redact",
    "sanitize",
    "install_global_filter",
    "install_import_hook",
]
