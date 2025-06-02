from __future__ import annotations

"""
Model initialization service for Saplings.

This module provides a service for initializing models.
"""

# Re-export the service from the services component
from saplings.services._internal.model_initialization_service import ModelInitializationService

__all__ = ["ModelInitializationService"]
