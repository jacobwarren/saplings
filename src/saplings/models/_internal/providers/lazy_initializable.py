from __future__ import annotations

"""
Lazy initializable interface for model adapters.

This module provides an interface for model adapters that support lazy initialization.
"""

# Re-export the interface from the models component
from saplings.models._internal.interfaces import LazyInitializable

__all__ = ["LazyInitializable"]
