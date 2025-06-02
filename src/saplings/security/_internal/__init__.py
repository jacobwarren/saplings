from __future__ import annotations

"""
Internal implementation of the Security module.
"""

# Import from subdirectories
from saplings.security._internal.hooks import (
    EXCLUDED_MODULES,
    INTERNAL_MODULE_PATTERN,
    INTERNAL_SYMBOL_PATTERN,
    PUBLIC_API_ALTERNATIVES,
    InternalAPIWarningFinder,
    install_import_hook,
)
from saplings.security._internal.logging import (
    RedactingFilter,
    install_global_filter,
    redact,
)
from saplings.security._internal.sanitization import (
    Sanitizer,
    sanitize,
    sanitize_prompt,
)

__all__ = [
    # Import hook
    "EXCLUDED_MODULES",
    "INTERNAL_MODULE_PATTERN",
    "INTERNAL_SYMBOL_PATTERN",
    "PUBLIC_API_ALTERNATIVES",
    "InternalAPIWarningFinder",
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
