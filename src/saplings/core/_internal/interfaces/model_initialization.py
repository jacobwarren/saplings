from __future__ import annotations

"""
Interface for model initialization service.

This module defines the interface for the model initialization service.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from saplings.core._internal.model_adapter import LLM


@dataclass
class ModelInitializationConfig:
    """Configuration for model initialization operations."""

    provider: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    additional_parameters: Dict[str, Any] = field(default_factory=dict)


class IModelInitializationService(Protocol):
    """Interface for model initialization service."""

    @abstractmethod
    async def get_model(self, timeout: Optional[float] = None) -> LLM:
        """
        Get the LLM instance.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            LLM: The initialized model

        Raises:
        ------
            ModelError: If the model is not initialized
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        ...

    @abstractmethod
    async def get_model_metadata(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get metadata about the model.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            Dict: Model metadata

        Raises:
        ------
            ModelError: If metadata cannot be retrieved
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        ...

    @abstractmethod
    async def initialize_model(
        self,
        config: ModelInitializationConfig,
        trace_id: Optional[str] = None,
    ) -> LLM:
        """
        Initialize a model with the specified configuration.

        Args:
        ----
            config: Model initialization configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            LLM: The initialized model

        Raises:
        ------
            ModelError: If initialization fails

        """
        ...
