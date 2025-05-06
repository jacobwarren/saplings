from __future__ import annotations

"""
Modality service interface for Saplings.

This module defines the interface for modality operations that handle
different input and output formats. This is a pure interface with no
implementation details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any


class IModalityService(ABC):
    """Interface for modality operations."""

    @abstractmethod
    def supported_modalities(self):
        """
        Get supported modalities.

        Returns
        -------
            List[str]: List of supported modality names

        """

    @abstractmethod
    async def format_output(
        self, content: Any, output_modality: str, trace_id: str | None = None
    ) -> Any:
        """
        Format output for a specific modality.

        Args:
        ----
            content: Content to format
            output_modality: Target modality
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Any: Formatted content

        """

    @abstractmethod
    async def process_input(
        self, content: Any, input_modality: str, trace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Process input from a specific modality.

        Args:
        ----
            content: Content to process
            input_modality: Source modality
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: Processed content

        """

    @abstractmethod
    async def convert(
        self,
        content: Any,
        source_modality: str,
        target_modality: str,
        trace_id: str | None = None,
    ) -> Any:
        """
        Convert content between modalities.

        Args:
        ----
            content: Content to convert
            source_modality: Source modality
            target_modality: Target modality
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Any: Converted content

        """
