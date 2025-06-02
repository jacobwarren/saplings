from __future__ import annotations

"""
Message module for Saplings.

This module defines the message classes used for communication with LLMs.
"""

from saplings.core._internal.message import (
    ContentType,
    FunctionCall,
    FunctionDefinition,
    Message,
    MessageContent,
    MessageRole,
)

__all__ = [
    "ContentType",
    "FunctionCall",
    "FunctionDefinition",
    "Message",
    "MessageContent",
    "MessageRole",
]
