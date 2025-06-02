from __future__ import annotations

"""
Service module for Graph-Aligned Sparse Attention (GASA).

This module provides a central service for managing GASA functionality.
It includes both the standard GASAService implementation and a NullGASAService
implementation for when GASA is disabled.
"""


from saplings.gasa._internal.service.gasa_service import GASAService
from saplings.gasa._internal.service.gasa_service_builder import GASAServiceBuilder
from saplings.gasa._internal.service.null_gasa_service import NullGASAService

__all__ = [
    "GASAService",
    "GASAServiceBuilder",
    "NullGASAService",
]
