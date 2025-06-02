from __future__ import annotations

"""
Security module for Saplings.

This module re-exports the public API from saplings.api.security.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides security utilities for Saplings, including:
- Sanitization to prevent injection attacks
- Log filtering to redact sensitive information
- Security validation utilities
- Import hooks for detecting usage of internal APIs
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.security.

# Import sanitize_prompt from internal module for backward compatibility
from saplings.security._internal.sanitization.sanitizer import sanitize_prompt

__all__ = [
    # Import hook
    "install_import_hook",
    # Log filter
    "RedactingFilter",
    "install_global_filter",
    "redact",
    # Sanitizer
    "Sanitizer",
    "sanitize",
    "sanitize_prompt",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__ and name != "sanitize_prompt":
        from saplings.api.security import (
            RedactingFilter,
            Sanitizer,
            install_global_filter,
            install_import_hook,
            redact,
            sanitize,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "RedactingFilter": RedactingFilter,
            "Sanitizer": Sanitizer,
            "install_global_filter": install_global_filter,
            "install_import_hook": install_import_hook,
            "redact": redact,
            "sanitize": sanitize,
        }

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
