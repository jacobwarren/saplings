from __future__ import annotations

"""
Retrieval Service API module for Saplings.

This module provides the retrieval service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.providers.retrieval_service import (
    RetrievalService as _RetrievalService,
)


@stable
class RetrievalService(_RetrievalService):
    """
    Service for retrieving documents.

    This service provides functionality for retrieving documents from memory,
    including semantic search and filtering.
    """


__all__ = [
    "RetrievalService",
]
