"""Platform-specific utilities."""

from __future__ import annotations

import importlib.util
import logging
import platform

logger = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine().startswith("arm")


def is_triton_available() -> bool:
    """Check if Triton is available."""
    if is_apple_silicon():
        return False

    try:
        if importlib.util.find_spec("triton") is not None:
            return True
        logger.warning("Triton not installed. Some GPU acceleration features may not be available.")
        return False
    except ImportError:
        logger.warning(
            "Could not check for Triton. Some GPU acceleration features may not be available."
        )
        return False
