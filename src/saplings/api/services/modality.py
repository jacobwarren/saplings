from __future__ import annotations

"""
Modality Service API module for Saplings.

This module provides the modality service implementation.
"""

from saplings.api.stability import beta
from saplings.services._internal.providers.modality_service import (
    ModalityService as _ModalityService,
)


@beta
class ModalityService(_ModalityService):
    """
    Service for handling different modalities.

    This service provides functionality for handling different modalities,
    including text, images, audio, and video.
    """


__all__ = [
    "ModalityService",
]
