from __future__ import annotations

"""
Model registry for managing LLM instances.

This module provides a registry for managing LLM instances to ensure
that only one instance of a model with the same configuration is created.
This helps reduce memory usage and improve performance.
"""


import hashlib
import logging
import weakref
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelKey(BaseModel):
    """
    Key for identifying a model in the registry.

    This class is used to generate a unique key for a model based on its
    provider, model name, and parameters.
    """

    provider: str
    model_name: str
    parameters: dict[str, Any] = {}

    def to_string(self):
        """
        Convert the model key to a string.

        Returns
        -------
            str: String representation of the model key

        """
        # Sort parameters to ensure consistent ordering
        sorted_params = dict(sorted(self.parameters.items()))

        # Create a string representation
        key_str = f"{self.provider}://{self.model_name}"

        # Add parameters if present
        if sorted_params:
            params_str = "&".join(f"{k}={v}" for k, v in sorted_params.items())
            key_str += f"?{params_str}"

        return key_str

    def to_hash(self):
        """
        Generate a hash of the model key.

        Returns
        -------
            str: Hash of the model key

        """
        key_str = self.to_string()
        return hashlib.md5(key_str.encode()).hexdigest()


class ModelRegistry:
    """
    Registry for managing LLM instances.

    This class ensures that only one instance of a model with the same
    configuration is created, which helps reduce memory usage and improve
    performance.
    """

    def __init__(self) -> None:
        """Initialize the model registry."""
        # Use weak references to allow models to be garbage collected
        # when they are no longer referenced elsewhere
        self._models = weakref.WeakValueDictionary()
        self._model_counts = {}
        logger.info("Model registry initialized")

    def register(self, model_key: ModelKey, model_instance: Any) -> None:
        """
        Register a model instance with the registry.

        Args:
        ----
            model_key: Key for the model
            model_instance: Model instance to register

        """
        key_hash = model_key.to_hash()
        self._models[key_hash] = model_instance

        # Update usage count
        if key_hash in self._model_counts:
            self._model_counts[key_hash] += 1
        else:
            self._model_counts[key_hash] = 1

        logger.info(
            "Registered model: %s (count: %d)", model_key.to_string(), self._model_counts[key_hash]
        )

    def get(self, model_key: ModelKey) -> Any | None:
        """
        Get a model instance from the registry.

        Args:
        ----
            model_key: Key for the model

        Returns:
        -------
            Optional[Any]: Model instance if found, None otherwise

        """
        key_hash = model_key.to_hash()
        model = self._models.get(key_hash)

        if model is not None:
            # Update usage count
            if key_hash in self._model_counts:
                self._model_counts[key_hash] += 1
            else:
                self._model_counts[key_hash] = 1

            logger.info(
                "Retrieved model: %s (count: %d)",
                model_key.to_string(),
                self._model_counts[key_hash],
            )

        return model

    def remove(self, model_key: ModelKey) -> None:
        """
        Remove a model instance from the registry.

        Args:
        ----
            model_key: Key for the model

        """
        key_hash = model_key.to_hash()
        if key_hash in self._models:
            del self._models[key_hash]

            # Update usage count
            if key_hash in self._model_counts:
                self._model_counts[key_hash] = max(0, self._model_counts[key_hash] - 1)

            logger.info(
                "Removed model: %s (count: %d)",
                model_key.to_string(),
                self._model_counts.get(key_hash, 0),
            )

    def clear(self) -> None:
        """Clear all model instances from the registry."""
        self._models.clear()
        self._model_counts.clear()
        logger.info("Cleared model registry")

    def get_model_count(self) -> int:
        """
        Get the number of model instances in the registry.

        Returns
        -------
            int: Number of model instances

        """
        return len(self._models)

    def get_model_keys(self) -> list[str]:
        """
        Get the keys of all model instances in the registry.

        Returns
        -------
            list: List of model keys

        """
        return list(self._models.keys())


def get_model_registry() -> ModelRegistry:
    """
    Get the model registry instance.

    This function is maintained for backward compatibility.
    New code should use constructor injection via the DI container.

    Returns
    -------
        ModelRegistry: Model registry instance from the DI container

    """
    from saplings.di import container

    return container.resolve(ModelRegistry)


def create_model_key(provider: str, model_name: str, **parameters: Any) -> ModelKey:
    """
    Create a model key from provider, model name, and parameters.

    Args:
    ----
        provider: Model provider
        model_name: Model name
        **parameters: Additional parameters

    Returns:
    -------
        ModelKey: Model key

    """
    return ModelKey(provider=provider, model_name=model_name, parameters=parameters)


# Function removed as part of ModelURI removal
