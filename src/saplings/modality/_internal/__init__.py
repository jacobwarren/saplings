from __future__ import annotations

"""
Internal implementation of the Modality module.

This module provides the implementation of modality components for the Saplings framework.
"""

# Import from subdirectories
from saplings.modality._internal.config import (
    ModalityConfig,
    ModalityType,
)
from saplings.modality._internal.handlers import (
    AudioHandler,
    ImageHandler,
    ModalityHandler,
    TextHandler,
    VideoHandler,
    get_handler_for_modality,
)
from saplings.modality._internal.interfaces import (
    IModalityHandler,
    IModalityHandlerRegistry,
)
from saplings.modality._internal.service import (
    ModalityService,
    ModalityServiceBuilder,
)

# Import from converters if needed
# from saplings.modality._internal.converters import ...

__all__ = [
    # Configuration
    "ModalityConfig",
    "ModalityType",
    # Handler implementations
    "AudioHandler",
    "ImageHandler",
    "ModalityHandler",
    "TextHandler",
    "VideoHandler",
    "get_handler_for_modality",
    # Service
    "ModalityService",
    "ModalityServiceBuilder",
    # Interfaces
    "IModalityHandler",
    "IModalityHandlerRegistry",
    # Converters (if needed)
]
