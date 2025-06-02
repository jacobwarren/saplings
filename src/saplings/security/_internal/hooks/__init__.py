from __future__ import annotations

"""
Hooks module for security components.

This module provides import hooks and other hook functionality for the Saplings framework.
"""

from saplings.security._internal.hooks.import_hook import (
    EXCLUDED_MODULES,
    INTERNAL_MODULE_PATTERN,
    INTERNAL_SYMBOL_PATTERN,
    PUBLIC_API_ALTERNATIVES,
    InternalAPIWarningFinder,
    install_import_hook,
)

__all__ = [
    "EXCLUDED_MODULES",
    "INTERNAL_MODULE_PATTERN",
    "INTERNAL_SYMBOL_PATTERN",
    "PUBLIC_API_ALTERNATIVES",
    "InternalAPIWarningFinder",
    "install_import_hook",
]
