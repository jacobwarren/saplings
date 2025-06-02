from __future__ import annotations

"""
Common registry functionality for Saplings.

This module provides common functionality shared between registry and service locator
to avoid circular dependencies.
"""

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global registry instance for early initialization
_global_registry: Optional[Any] = None
_registry_lock = threading.RLock()

# Global service locator instance
_service_locator = None


def get_service_locator():
    """
    Get the service locator instance.

    This function is used by both registry and service_locator modules
    to avoid circular imports.

    Returns
    -------
        The service locator instance

    """
    global _service_locator

    if _service_locator is None:
        # Lazy import to avoid circular imports
        from saplings.registry._internal.service.service_locator import ServiceLocator

        _service_locator = ServiceLocator.get_instance()

    return _service_locator
