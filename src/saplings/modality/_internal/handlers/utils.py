from __future__ import annotations

"""
Utility functions for modality handlers.

This module provides utility functions for working with modality handlers.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saplings.api.models import LLM
    from saplings.modality._internal.handlers.modality_handler import ModalityHandler


def get_handler_for_modality(modality: str, model: "LLM") -> "ModalityHandler":
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
    # Import handlers here to avoid circular imports
    from saplings.modality._internal.handlers.audio_handler import AudioHandler
    from saplings.modality._internal.handlers.image_handler import ImageHandler
    from saplings.modality._internal.handlers.text_handler import TextHandler
    from saplings.modality._internal.handlers.video_handler import VideoHandler

    handlers = {
        "text": TextHandler,
        "image": ImageHandler,
        "audio": AudioHandler,
        "video": VideoHandler,
    }

    if modality not in handlers:
        msg = f"Unsupported modality: {modality}"
        raise ValueError(msg)

    return handlers[modality](model)
