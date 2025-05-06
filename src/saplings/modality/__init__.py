from __future__ import annotations

"""
Modality module for Saplings.

This module provides support for different modalities (text, image, audio, video)
that can be used by agents to process and generate content in different formats.
"""


from saplings.modality.handlers import (
    AudioHandler,
    ImageHandler,
    ModalityHandler,
    TextHandler,
    VideoHandler,
    get_handler_for_modality,
)

__all__ = [
    "AudioHandler",
    "ImageHandler",
    "ModalityHandler",
    "TextHandler",
    "VideoHandler",
    "get_handler_for_modality",
]
