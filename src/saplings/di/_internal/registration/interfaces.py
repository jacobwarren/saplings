from __future__ import annotations

"""
Interface definitions for the DI container.

This module defines interfaces used by the DI container to avoid circular imports.
"""

from typing import Generic, TypeVar

T = TypeVar("T")


class Provider(Generic[T]):
    """Base provider interface."""

    def get(self) -> T:
        """Get the instance."""
        raise NotImplementedError


__all__ = ["Provider"]
