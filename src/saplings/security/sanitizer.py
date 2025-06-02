from __future__ import annotations

"""
Prompt sanitization module for Saplings.

This module provides utilities for sanitizing prompts to prevent prompt injection.
This module is deprecated and will be removed in a future version.
Please use saplings.api.security.sanitize_prompt instead.
"""

from saplings.security._internal.sanitization.sanitizer import sanitize_prompt

__all__ = ["sanitize_prompt"]
