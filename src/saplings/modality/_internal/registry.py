from __future__ import annotations

"""
Registry for modality handlers.

This module provides a registry for modality handlers, allowing them to be
registered and retrieved by modality type.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type

from saplings.modality._internal.handlers import (
    AudioHandler,
    ImageHandler,
    ModalityHandler,
    TextHandler,
    VideoHandler,
)
from saplings.modality._internal.interfaces import IModalityHandlerRegistry

logger = logging.getLogger(__name__)


class ModalityHandlerRegistry(IModalityHandlerRegistry):
    """
    Registry for modality handlers.

    This class provides a registry for modality handlers, allowing them to be
    registered and retrieved by modality type. It supports lazy initialization
    of handlers to improve performance and reduce memory usage.
    """

    def __init__(self) -> None:
        """Initialize the modality handler registry."""
        self._handler_factories: Dict[str, Callable[..., ModalityHandler]] = {}
        self._handler_classes: Dict[str, Type[ModalityHandler]] = {
            "text": TextHandler,
            "image": ImageHandler,
            "audio": AudioHandler,
            "video": VideoHandler,
        }

    def register_handler_class(self, modality: str, handler_class: Type[ModalityHandler]) -> None:
        """
        Register a handler class for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            handler_class: Handler class to register

        """
        self._handler_classes[modality] = handler_class
        logger.debug(f"Registered handler class for modality: {modality}")

    def register_handler_factory(
        self, modality: str, factory: Callable[..., ModalityHandler]
    ) -> None:
        """
        Register a handler factory for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            factory: Factory function to create handler instances

        """
        self._handler_factories[modality] = factory
        logger.debug(f"Registered handler factory for modality: {modality}")

    def get_handler(self, modality: str, model: Any) -> ModalityHandler:
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
        # Check if we have a factory for this modality
        if modality in self._handler_factories:
            return self._handler_factories[modality](model)

        # Check if we have a class for this modality
        if modality in self._handler_classes:
            return self._handler_classes[modality](model)

        # Modality not supported
        msg = f"Unsupported modality: {modality}"
        raise ValueError(msg)

    def supported_modalities(self) -> list[str]:
        """
        Get list of supported modalities.

        Returns
        -------
            list[str]: List of supported modality names

        """
        # Combine modalities from factories and classes
        return list(set(self._handler_factories.keys()) | set(self._handler_classes.keys()))


# Global registry instance
_registry: Optional[ModalityHandlerRegistry] = None


def get_modality_handler_registry() -> ModalityHandlerRegistry:
    """
    Get the global modality handler registry.

    Returns
    -------
        ModalityHandlerRegistry: Global modality handler registry

    """
    global _registry
    if _registry is None:
        _registry = ModalityHandlerRegistry()
    return _registry
