from __future__ import annotations

"""
Sanitization module for Saplings.

This module provides utilities for sanitizing inputs to prevent injection attacks.
"""


import re
from typing import Any, Dict, List, Optional, Set, Union


class Sanitizer:
    """
    Sanitizer for preventing prompt injection attacks.

    This class provides methods for sanitizing inputs to prevent
    prompt injection attacks and other security issues.
    """

    # Default patterns to detect potential prompt injections
    DEFAULT_INJECTION_PATTERNS = [
        # System prompt injections
        r"(system:|system prompt:|<\|system\|>)",
        # Instruction injections
        r"(ignore previous instructions|ignore all instructions|disregard previous instructions)",
        # Role injections
        r"(you are now|you're now|you will now act as|pretend you are|you are a|you're a)",
        # Delimiter injections
        r"(```|---|===|<<<|>>>|\*\*\*|\+\+\+)",
        # Command injections
        r"(sudo|rm -rf|chmod|chown|wget|curl|bash|sh|exec|eval|system|popen|subprocess)",
        # SQL injections
        r"(SELECT.*FROM|INSERT.*INTO|UPDATE.*SET|DELETE.*FROM|DROP.*TABLE|ALTER.*TABLE)",
        # Path traversal
        r"(\.\./|\.\./\.\./|/etc/passwd|/etc/shadow|/proc/self)",
        # XML/HTML injections
        r"(<script>|</script>|<iframe>|</iframe>|<img|<svg|<xml|<!ENTITY)",
    ]

    def __init__(
        self,
        injection_patterns: Optional[List[str]] = None,
        allowed_tags: Optional[Set[str]] = None,
        max_input_length: int = 32768,
    ) -> None:
        """
        Initialize the sanitizer.

        Args:
        ----
            injection_patterns: List of regex patterns to detect injections.
                If None, uses default patterns.
            allowed_tags: Set of HTML/XML tags that are allowed.
                If None, all tags are removed.
            max_input_length: Maximum allowed input length.

        """
        self.injection_patterns = injection_patterns or self.DEFAULT_INJECTION_PATTERNS
        self.allowed_tags = allowed_tags or set()
        self.max_input_length = max_input_length

        # Compile patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns
        ]

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input.

        Args:
        ----
            text: Text to sanitize

        Returns:
        -------
            str: Sanitized text

        Raises:
        ------
            ValueError: If the input contains potential injection patterns

        """
        # Check input length
        if len(text) > self.max_input_length:
            msg = f"Input length ({len(text)}) exceeds maximum allowed length ({self.max_input_length})"
            raise ValueError(msg)

        # Check for potential injections
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                msg = f"Input contains potential injection pattern: {match.group(0)}"
                raise ValueError(msg)

        # Remove HTML/XML tags
        if not self.allowed_tags:
            # Remove all tags
            text = re.sub(r"<[^>]*>", "", text)
        else:
            # Remove disallowed tags
            for tag in re.findall(r"</?([a-zA-Z0-9]+)[^>]*>", text):
                if tag.lower() not in self.allowed_tags:
                    text = re.sub(f"</?{tag}[^>]*>", "", text)

        return text

    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize dictionary input.

        Args:
        ----
            data: Dictionary to sanitize

        Returns:
        -------
            Dict[str, Any]: Sanitized dictionary

        """
        result = {}
        for key, value in data.items():
            # Sanitize the key
            key = self.sanitize_text(str(key))

            # Sanitize the value based on its type
            if isinstance(value, str):
                result[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = self.sanitize_list(value)
            else:
                # Keep other types as is
                result[key] = value

        return result

    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """
        Sanitize list input.

        Args:
        ----
            data: List to sanitize

        Returns:
        -------
            List[Any]: Sanitized list

        """
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(self.sanitize_text(item))
            elif isinstance(item, dict):
                result.append(self.sanitize_dict(item))
            elif isinstance(item, list):
                result.append(self.sanitize_list(item))
            else:
                # Keep other types as is
                result.append(item)

        return result


def sanitize(
    input_data: Union[str, Dict[str, Any], List[Any]],
    injection_patterns: Optional[List[str]] = None,
    allowed_tags: Optional[Set[str]] = None,
    max_input_length: int = 32768,
) -> Union[str, Dict[str, Any], List[Any]]:
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
        Union[str, Dict[str, Any], List[Any]]: Sanitized data

    """
    sanitizer = Sanitizer(
        injection_patterns=injection_patterns,
        allowed_tags=allowed_tags,
        max_input_length=max_input_length,
    )

    if isinstance(input_data, str):
        return sanitizer.sanitize_text(input_data)
    if isinstance(input_data, dict):
        return sanitizer.sanitize_dict(input_data)
    if isinstance(input_data, list):
        return sanitizer.sanitize_list(input_data)

    # Return other types as is
    return input_data


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize a prompt to prevent injection attacks.

    Args:
    ----
        prompt: The prompt to sanitize

    Returns:
    -------
        The sanitized prompt

    """
    result = sanitize(prompt)
    if isinstance(result, str):
        return result
    # This should never happen since we're passing a string
    return str(result)
