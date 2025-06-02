from __future__ import annotations

"""
Lazy initializable interface for model adapters.

This module provides an interface for model adapters that support lazy initialization.
It re-exports the LazyInitializable protocol from the models component.
"""

# Re-export the interface from the models component
from saplings.models._internal.interfaces import LazyInitializable

__all__ = ["LazyInitializable"]
