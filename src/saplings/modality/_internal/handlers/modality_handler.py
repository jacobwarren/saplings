from __future__ import annotations

"""
Base modality handler for Saplings.

This module provides the base handler class for different modalities.
"""

import logging
from typing import Any

from saplings.api.models import LLM
from saplings.core.message import MessageContent
from saplings.modality._internal.interfaces import IModalityHandler

logger = logging.getLogger(__name__)


class ModalityHandler(IModalityHandler):
    """Base class for modality handlers."""

    def __init__(self, model: LLM) -> None:
        """
        Initialize the modality handler.

        Args:
        ----
            model: LLM model to use for processing

        """
        self.model = model

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
        msg = "Subclasses must implement process_input"
        raise NotImplementedError(msg)

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
        msg = "Subclasses must implement format_output"
        raise NotImplementedError(msg)

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
        msg = "Subclasses must implement to_message_content"
        raise NotImplementedError(msg)

    @classmethod
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
        msg = "Subclasses must implement from_message_content"
        raise NotImplementedError(msg)
