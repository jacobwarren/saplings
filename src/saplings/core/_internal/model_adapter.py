from __future__ import annotations

"""
Model adapter module for Saplings.

This module provides a unified interface for interacting with different LLM providers.
It includes a model registry to ensure only one instance of a model with the same
configuration is created, which helps reduce memory usage and improve performance.
"""

import importlib
import logging
from typing import cast

from saplings.core._internal.model_interface import LLM
from saplings.models._internal.interfaces import ModelAdapterFactory

logger = logging.getLogger(__name__)


def initialize_model_factory() -> None:
    """
    Initialize the model adapter factory.

    This function sets up the model adapter factory for the LLM class.
    It should be called during application initialization.
    """
    try:
        # Import the factory dynamically to avoid circular imports
        factory_module = importlib.import_module(
            "saplings.models._internal.providers.model_adapter_factory"
        )
        factory_class = factory_module.ModelAdapterFactory

        # Set the factory on the LLM class
        LLM.set_factory(cast(ModelAdapterFactory, factory_class))
        logger.debug("Model adapter factory initialized successfully")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to initialize model adapter factory: {e}")
        raise ImportError(f"Failed to initialize model adapter factory: {e}") from e


# Initialize the factory when this module is imported
# Commented out to avoid circular imports during package initialization
# initialize_model_factory()
