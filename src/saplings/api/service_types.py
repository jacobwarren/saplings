from __future__ import annotations

"""
Service types module for Saplings.

This module provides common types and interfaces for services to avoid circular imports.
"""

from typing import Protocol

from saplings.api.stability import stable


@stable
class Service(Protocol):
    """Base protocol for all services."""

    @property
    def name(self) -> str:
        """Get the name of the service."""
        ...


__all__ = ["Service"]
