from __future__ import annotations

"""
Service module for modality components.

This module provides service functionality for modalities in the Saplings framework.
"""

from saplings.modality._internal.service.modality_service import ModalityService
from saplings.modality._internal.service.modality_service_builder import ModalityServiceBuilder

__all__ = [
    "ModalityService",
    "ModalityServiceBuilder",
]
