from __future__ import annotations

"""
Modality module for Saplings.

This module re-exports the public API from saplings.api.modality.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides support for different modalities (text, image, audio, video)
that can be used by agents to process and generate content in different formats.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.modality.

__all__ = [
    "AudioHandler",
    "ImageHandler",
    "ModalityHandler",
    "ModalityConfig",
    "ModalityType",
    "TextHandler",
    "VideoHandler",
    "get_handler_for_modality",
    "ModalityHandlerRegistry",
    "get_modality_handler_registry",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.modality import (
            AudioHandler,
            ImageHandler,
            ModalityConfig,
            ModalityHandler,
            ModalityHandlerRegistry,
            ModalityType,
            TextHandler,
            VideoHandler,
            get_handler_for_modality,
            get_modality_handler_registry,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "AudioHandler": AudioHandler,
            "ImageHandler": ImageHandler,
            "ModalityHandler": ModalityHandler,
            "ModalityConfig": ModalityConfig,
            "ModalityType": ModalityType,
            "TextHandler": TextHandler,
            "VideoHandler": VideoHandler,
            "get_handler_for_modality": get_handler_for_modality,
            "ModalityHandlerRegistry": ModalityHandlerRegistry,
            "get_modality_handler_registry": get_modality_handler_registry,
        }

        # Return the requested attribute
        return globals_dict[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
