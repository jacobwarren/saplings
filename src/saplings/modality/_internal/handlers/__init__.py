from __future__ import annotations

"""
Handlers module for modality components.

This module provides handler implementations for different modalities in the Saplings framework.
"""

from saplings.modality._internal.handlers.audio_handler import AudioHandler
from saplings.modality._internal.handlers.image_handler import ImageHandler
from saplings.modality._internal.handlers.modality_handler import ModalityHandler
from saplings.modality._internal.handlers.text_handler import TextHandler
from saplings.modality._internal.handlers.utils import get_handler_for_modality
from saplings.modality._internal.handlers.video_handler import VideoHandler

__all__ = [
    "AudioHandler",
    "ImageHandler",
    "ModalityHandler",
    "TextHandler",
    "VideoHandler",
    "get_handler_for_modality",
]
