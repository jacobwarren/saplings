from __future__ import annotations

"""
Common type definitions for Saplings.

This module provides common type definitions used across the Saplings framework.
"""

from typing import Any, Dict, List, Union

# JSON-related types
JsonDict = Dict[str, Any]
JsonValue = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]

__all__ = [
    "JsonDict",
    "JsonValue",
]
