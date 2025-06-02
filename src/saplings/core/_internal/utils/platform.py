"""Platform-specific utilities."""

from __future__ import annotations

import importlib.util
import logging
import platform
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine().startswith("arm")


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def get_platform_info() -> Dict[str, Any]:
    """
    Get information about the current platform.

    Returns
    -------
        Dict containing platform information

    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "is_macos": is_macos(),
        "is_windows": is_windows(),
        "is_linux": is_linux(),
        "is_apple_silicon": is_apple_silicon(),
    }


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
