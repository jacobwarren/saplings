from __future__ import annotations

"""
Prompt sanitization utilities for Saplings.

This module provides functions to sanitize prompts to prevent various
injection and attack vectors.
"""


import re
from re import Pattern

# Regular expressions for sanitization
URL_RE = re.compile(r"https?://\S+")
SHELL_RE = re.compile(r"[;&|`$()<>]")
# Add more patterns for other potential attack vectors
SQL_INJECTION_RE = re.compile(
    r"(?i)(?:(?:select|update|delete|insert|drop|alter).*?(?:from|into|where|table))"
)
PATH_TRAVERSAL_RE = re.compile(r"\.\.\/")
HTML_TAGS_RE = re.compile(r"</?[a-z]+.*?>", re.IGNORECASE)
SCRIPT_TAGS_RE = re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL)


def sanitize_prompt(
    raw: str,
    max_len: int = 8_192,
    remove_urls: bool = True,
    remove_shell_metacharacters: bool = True,
    remove_sql_patterns: bool = True,
    remove_path_traversal: bool = True,
    remove_html: bool = True,
    custom_patterns: list[Pattern[str]] | None = None,
) -> str:
    """
    Remove dangerous substrings and truncate at max_len.

    Args:
    ----
        raw: The raw input string to sanitize
        max_len: Maximum length of the output string
        remove_urls: Whether to redact URLs
        remove_shell_metacharacters: Whether to remove shell metacharacters
        remove_sql_patterns: Whether to remove SQL injection patterns
        remove_path_traversal: Whether to remove path traversal patterns
        remove_html: Whether to remove HTML and script tags
        custom_patterns: Additional custom patterns to remove

    Returns:
    -------
        Sanitized string

    """
    if not raw:
        return ""

    clean = raw

    # Apply sanitization filters based on parameters
    if remove_urls:
        clean = URL_RE.sub("[URL]", clean)

    if remove_shell_metacharacters:
        clean = SHELL_RE.sub("", clean)

    if remove_sql_patterns:
        clean = SQL_INJECTION_RE.sub("[SQL]", clean)

    if remove_path_traversal:
        clean = PATH_TRAVERSAL_RE.sub("", clean)

    if remove_html:
        clean = HTML_TAGS_RE.sub("", clean)
        clean = SCRIPT_TAGS_RE.sub("", clean)

    # Apply any custom patterns
    if custom_patterns:
        for pattern in custom_patterns:
            clean = pattern.sub("", clean)

    # Truncate to max_len
    return clean[:max_len]


def sanitize_document_collection(documents: list[str], max_len: int = 8_192, **kwargs) -> list[str]:
    """
    Sanitize a collection of documents.

    Args:
    ----
        documents: List of strings to sanitize
        max_len: Maximum length of each document
        **kwargs: Additional arguments to pass to sanitize_prompt

    Returns:
    -------
        List of sanitized strings

    """
    return [sanitize_prompt(doc, max_len=max_len, **kwargs) for doc in documents]
