from __future__ import annotations

"""Plugin modules for the Saplings framework."""

import logging

# Import from the public API
from saplings.api.plugins import (
    CodeIndexer,
    CodeValidator,
    FactualValidator,
    SecureMemoryStore,
)

# Import register_all_plugins from internal implementation
from saplings.plugins._internal.loader import register_all_plugins

# Set up logging
logger = logging.getLogger(__name__)

# Don't register plugins automatically during import
# This prevents side effects and circular dependencies
# Users should call register_all_plugins() explicitly when needed
logger.debug("Plugin classes imported successfully")

__all__ = [
    "CodeIndexer",
    "CodeValidator",
    "FactualValidator",
    "SecureMemoryStore",
    "register_all_plugins",
]
