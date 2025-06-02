from __future__ import annotations

"""
Interface definitions for modality handlers.

This module defines the interfaces for modality handlers to avoid circular imports.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

from saplings.core.message import MessageContent

logger = logging.getLogger(__name__)


class IModalityHandler(ABC):
    """Base interface for modality handlers."""

    @abstractmethod
    async def process_input(self, input_data: Any) -> Any:
        """
        Process input data for this modality.

        Args:
        ----
            input_data: Input data to process

        Returns:
        -------
            Processed input data

        """

    @abstractmethod
    async def format_output(self, output: Any) -> Any:
        """
        Format output data for this modality.

        Args:
        ----
            output: Output data to format

        Returns:
        -------
            Formatted output data

        """

    @abstractmethod
    def to_message_content(self, data: Any) -> MessageContent:
        """
        Convert data to MessageContent.

        Args:
        ----
            data: Data to convert

        Returns:
        -------
            MessageContent object

        """

    @classmethod
    @abstractmethod
    def from_message_content(cls, content: MessageContent) -> Any:
        """
        Convert MessageContent to data.

        Args:
        ----
            content: MessageContent to convert

        Returns:
        -------
            Converted data

        """


class IModalityHandlerRegistry(Protocol):
    """Interface for modality handler registry."""

    def register_handler_class(self, modality: str, handler_class: Any) -> None:
        """
        Register a handler class for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            handler_class: Handler class to register

        """

    def register_handler_factory(self, modality: str, factory: Any) -> None:
        """
        Register a handler factory for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            factory: Factory function to create handler instances

        """

    def get_handler(self, modality: str, model: Any) -> Any:
        """
        Get a handler for a specific modality.

        Args:
        ----
            modality: Modality name (text, image, audio, video, etc.)
            model: LLM model to use for processing

        Returns:
        -------
            ModalityHandler: Handler for the specified modality

        Raises:
        ------
            ValueError: If the modality is not supported

        """

    def supported_modalities(self) -> list[str]:
        """
        Get list of supported modalities.

        Returns
        -------
            list[str]: List of supported modality names

        """
