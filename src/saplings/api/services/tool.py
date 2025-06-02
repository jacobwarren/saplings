from __future__ import annotations

"""
Tool Service API module for Saplings.

This module provides the tool service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.providers.tool_service import ToolService as _ToolService


@stable
class ToolService(_ToolService):
    """
    Service for managing tools.

    This service provides functionality for managing tools, including
    registering tools, creating dynamic tools, and executing tools.
    """


__all__ = [
    "ToolService",
]
