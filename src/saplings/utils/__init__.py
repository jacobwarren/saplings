from __future__ import annotations

"""
Utility functions for Saplings.

This package contains utility functions that are used throughout the Saplings codebase.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.utils.

__all__ = [
    # Synchronous utilities
    "get_model_sync",
    "run_sync",
    # Asynchronous utilities
    "async_run_sync",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.utils import async_run_sync, get_model_sync, run_sync

        # Create a mapping of names to their values
        globals_dict = {
            "async_run_sync": async_run_sync,
            "get_model_sync": get_model_sync,
            "run_sync": run_sync,
        }

        # Return the requested attribute
        return globals_dict[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
