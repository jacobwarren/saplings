from __future__ import annotations

"""
Text modality handler for Saplings.

This module provides the handler implementation for text modality.
"""

from typing import Any

from saplings.core.message import ContentType, MessageContent
from saplings.modality._internal.handlers.modality_handler import ModalityHandler


class TextHandler(ModalityHandler):
    """Handler for text modality."""

    async def process_input(self, input_data: Any) -> str:
        """
        Process text input.

        Args:
        ----
            input_data: Input text data

        Returns:
        -------
            Processed text

        """
        if isinstance(input_data, str):
            return input_data
        return str(input_data)

    async def format_output(self, output: Any) -> str:
        """
        Format output as text.

        Args:
        ----
            output: Output data

        Returns:
        -------
            Formatted text

        """
        if isinstance(output, str):
            return output
        return str(output)

    def to_message_content(self, data: str) -> MessageContent:
        """
        Convert text data to MessageContent.

        Args:
        ----
            data: Text data

        Returns:
        -------
            MessageContent object

        """
        return MessageContent(
            type=ContentType.TEXT,
            text=data,
            image_url=None,
            image_data=None,
            audio_url=None,
            audio_data=None,
            video_url=None,
            video_data=None,
        )

    @classmethod
    def from_message_content(cls, content: MessageContent) -> str:
        """
        Convert MessageContent to text data.

        Args:
        ----
            content: MessageContent to convert

        Returns:
        -------
            Text data

        """
        if content.type != ContentType.TEXT:
            msg = f"Expected TEXT content type, got {content.type}"
            raise ValueError(msg)
        return content.text or ""
