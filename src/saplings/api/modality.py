from __future__ import annotations

"""
Public API for modality handling.

This module provides the public API for modality handling, including:
- Modality handlers for text, image, audio, and video
- Configuration for modality handling
- Registry for modality handlers
- Utilities for working with different modalities
"""

from typing import Any, Callable

from saplings.api.stability import beta
from saplings.modality._internal import (
    AudioHandler as _AudioHandler,
)
from saplings.modality._internal import (
    ImageHandler as _ImageHandler,
)
from saplings.modality._internal import (
    ModalityConfig as _ModalityConfig,
)
from saplings.modality._internal import (
    ModalityHandler as _ModalityHandler,
)
from saplings.modality._internal import (
    ModalityService as _ModalityService,
)
from saplings.modality._internal import (
    ModalityServiceBuilder as _ModalityServiceBuilder,
)
from saplings.modality._internal import (
    ModalityType as _ModalityType,
)
from saplings.modality._internal import (
    TextHandler as _TextHandler,
)
from saplings.modality._internal import (
    VideoHandler as _VideoHandler,
)
from saplings.modality._internal import (
    get_handler_for_modality as _get_handler_for_modality,
)
from saplings.modality._internal.registry import (
    get_modality_handler_registry as _get_modality_handler_registry,
)


@beta
class ModalityHandler(_ModalityHandler):
    """Base class for modality handlers."""


@beta
class TextHandler(_TextHandler):
    """Handler for text modality."""


@beta
class ImageHandler(_ImageHandler):
    """Handler for image modality."""


@beta
class AudioHandler(_AudioHandler):
    """Handler for audio modality."""


@beta
class VideoHandler(_VideoHandler):
    """Handler for video modality."""


# Re-export the ModalityType enum directly
ModalityType = beta(_ModalityType)


@beta
class ModalityConfig(_ModalityConfig):
    """Configuration for modality handling."""


@beta
class ModalityHandlerRegistry:
    """
    Registry for modality handlers.

    This class provides a registry for modality handlers, allowing them to be
    registered and retrieved by modality type. It supports lazy initialization
    of handlers to improve performance and reduce memory usage.
    """

    def __init__(self) -> None:
        """Initialize the modality handler registry."""
        self._registry = _get_modality_handler_registry()

    def register_handler_class(self, modality: str, handler_class: Any) -> None:
        """
        Register a handler class for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            handler_class: Handler class to register

        """
        self._registry.register_handler_class(modality, handler_class)

    def register_handler_factory(self, modality: str, factory: Callable[..., Any]) -> None:
        """
        Register a handler factory for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            factory: Factory function to create handler instances

        """
        self._registry.register_handler_factory(modality, factory)

    def get_handler(self, modality: str, model: Any) -> Any:
        """
        Get a handler for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            model: LLM model to use for processing

        Returns:
        -------
            ModalityHandler: Handler for the specified modality

        Raises:
        ------
            ValueError: If the modality is not supported

        """
        return self._registry.get_handler(modality, model)

    def supported_modalities(self) -> list[str]:
        """
        Get list of supported modalities.

        Returns
        -------
            list[str]: List of supported modality names

        """
        return self._registry.supported_modalities()


@beta
def get_modality_handler_registry() -> ModalityHandlerRegistry:
    """
    Get the global modality handler registry.

    Returns
    -------
        ModalityHandlerRegistry: Global modality handler registry

    """
    # Create a new public API registry that wraps the internal registry
    return ModalityHandlerRegistry()


@beta
def get_handler_for_modality(modality: str, model: Any) -> Any:
    """
    Get a handler for a specific modality.

    Args:
    ----
        modality: Modality name (text, image, audio, video)
        model: LLM model to use for processing

    Returns:
    -------
        ModalityHandler: Handler for the specified modality

    Raises:
    ------
        ValueError: If the modality is not supported

    """
    handler = _get_handler_for_modality(modality, model)

    # Convert the internal handler to the public API handler
    if isinstance(handler, _TextHandler):
        return TextHandler(model)
    elif isinstance(handler, _ImageHandler):
        return ImageHandler(model)
    elif isinstance(handler, _AudioHandler):
        return AudioHandler(model)
    elif isinstance(handler, _VideoHandler):
        return VideoHandler(model)
    else:
        return ModalityHandler(model)


@beta
class ModalityService(_ModalityService):
    """
    Service that manages multimodal capabilities.

    This service provides functionality for processing and converting content
    between different modalities (text, image, audio, video).
    """


@beta
class ModalityServiceBuilder(_ModalityServiceBuilder):
    """
    Builder for ModalityService.

    This class provides a fluent interface for building ModalityService instances with
    proper configuration and dependency injection.
    """


__all__ = [
    # Handlers
    "AudioHandler",
    "ImageHandler",
    "ModalityHandler",
    "TextHandler",
    "VideoHandler",
    # Configuration
    "ModalityConfig",
    "ModalityType",
    # Registry
    "ModalityHandlerRegistry",
    "get_modality_handler_registry",
    # Utilities
    "get_handler_for_modality",
    # Service
    "ModalityService",
    "ModalityServiceBuilder",
]
