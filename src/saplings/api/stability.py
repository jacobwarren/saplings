from __future__ import annotations

"""
API Stability Annotations for Saplings.

This module provides annotations for indicating the stability level of API components.
These annotations help users understand which parts of the API are stable and can be
relied upon, and which parts may change in future versions.

Stability levels:
- STABLE: The API is stable and will not change in a backward-incompatible way within the same major version.
- BETA: The API is mostly stable but may change in minor ways in future versions.
- ALPHA: The API is experimental and may change significantly in future versions.
- INTERNAL: The API is for internal use only and may change without notice.
"""

import enum
import sys
from typing import Any, Callable, Optional, TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


# We need to define a simple decorator for StabilityLevel to avoid circular imports
def _mark_stable(cls):
    """Mark a class as stable without using the stable decorator."""
    cls.__doc__ = f"{cls.__doc__}\n\nStability: Stable"
    return cls


@_mark_stable
class StabilityLevel(str, enum.Enum):
    """
    Stability level for API components.

    This enum is considered stable and will not change in a backward-incompatible way.
    """

    STABLE = "stable"
    """
    The API is stable and will not change in a backward-incompatible way within the same major version.

    Changes to stable APIs will follow semantic versioning:
    - Major version changes may include breaking changes
    - Minor version changes will be backward-compatible
    - Patch version changes will be backward-compatible bug fixes
    """

    BETA = "beta"
    """
    The API is mostly stable but may change in minor ways in future versions.

    Beta APIs are expected to eventually become stable, but may still undergo
    minor changes based on user feedback.
    """

    ALPHA = "alpha"
    """
    The API is experimental and may change significantly in future versions.

    Alpha APIs are early previews of new functionality and may change
    significantly based on user feedback and implementation experience.
    """

    INTERNAL = "internal"
    """
    The API is for internal use only and may change without notice.

    Internal APIs are not intended for use outside of the Saplings codebase
    and may change without notice.
    """


# Set the stability level for StabilityLevel itself
StabilityLevel.__stability__ = StabilityLevel.STABLE


def stability(level: StabilityLevel) -> Callable[[T], T]:
    """
    Decorator for annotating API components with stability information.

    Args:
    ----
        level: The stability level of the API component

    Returns:
    -------
        A decorator that adds stability information to the API component

    """

    def decorator(obj: T) -> T:
        # Store stability level in the object's metadata
        obj.__stability__ = level

        # Update the docstring to include stability information
        if obj.__doc__:
            obj.__doc__ = f"{obj.__doc__}\n\nStability: {level.value.capitalize()}"
        else:
            obj.__doc__ = f"Stability: {level.value.capitalize()}"

        return obj

    return decorator


def get_stability(obj: Any) -> Optional[StabilityLevel]:
    """
    Get the stability level of an API component.

    Args:
    ----
        obj: The API component to check

    Returns:
    -------
        The stability level of the API component, or None if not annotated

    """
    return getattr(obj, "__stability__", None)


def is_stable(obj: Any) -> bool:
    """
    Check if an API component is stable.

    Args:
    ----
        obj: The API component to check

    Returns:
    -------
        True if the API component is stable, False otherwise

    """
    return get_stability(obj) == StabilityLevel.STABLE


def is_public(obj: Any) -> bool:
    """
    Check if an API component is public (stable, beta, or alpha).

    Args:
    ----
        obj: The API component to check

    Returns:
    -------
        True if the API component is public, False otherwise

    """
    stability = get_stability(obj)
    return stability in (StabilityLevel.STABLE, StabilityLevel.BETA, StabilityLevel.ALPHA)


# Convenience decorators for each stability level
def stable(obj: T) -> T:
    """Mark an API component as stable."""
    return stability(StabilityLevel.STABLE)(obj)


def beta(obj: T) -> T:
    """Mark an API component as beta."""
    return stability(StabilityLevel.BETA)(obj)


def alpha(obj: T) -> T:
    """Mark an API component as alpha."""
    return stability(StabilityLevel.ALPHA)(obj)


# Alias for alpha for backward compatibility
def experimental(obj: T) -> T:
    """Mark an API component as experimental (alias for alpha)."""
    return alpha(obj)


def internal(obj: T) -> T:
    """Mark an API component as internal."""
    return stability(StabilityLevel.INTERNAL)(obj)


__all__ = [
    # Stability level enum
    "StabilityLevel",
    # Main decorator
    "stability",
    # Convenience decorators
    "stable",
    "beta",
    "alpha",
    "experimental",
    "internal",
    # Utility functions
    "get_stability",
    "is_stable",
    "is_public",
]
