from __future__ import annotations

"""
Model initialization service builder for Saplings.

This module provides a builder for the model initialization service.
"""

# Re-export the builder from the services component
from saplings.services._internal.builders.model_initialization_service_builder import (
    ModelInitializationServiceBuilder,
)

__all__ = ["ModelInitializationServiceBuilder"]
