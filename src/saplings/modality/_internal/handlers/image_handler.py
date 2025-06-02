from __future__ import annotations

"""
Image modality handler for Saplings.

This module provides the handler implementation for image modality.
"""

import os
from typing import Any

from saplings.core.message import ContentType, MessageContent
from saplings.modality._internal.handlers.modality_handler import ModalityHandler


class ImageHandler(ModalityHandler):
    """Handler for image modality."""

    async def process_input(self, input_data: Any) -> dict[str, Any]:
        """
        Process image input.

        Args:
        ----
            input_data: Input image data (URL, file path, or bytes)

        Returns:
        -------
            Processed image data

        """
        if isinstance(input_data, str):
            # Check if it's a URL or file path
            if input_data.startswith(("http://", "https://", "data:")):
                return {"type": "url", "url": input_data}
            if os.path.exists(input_data):
                # Read file content
                with open(input_data, "rb") as f:
                    image_data = f.read()
                return {"type": "data", "data": image_data}
            msg = f"Invalid image path or URL: {input_data}"
            raise ValueError(msg)
        if isinstance(input_data, bytes):
            return {"type": "data", "data": input_data}
        if isinstance(input_data, dict) and ("url" in input_data or "data" in input_data):
            return input_data
        msg = f"Unsupported image input type: {type(input_data)}"
        raise ValueError(msg)

    async def format_output(self, output: Any) -> dict[str, Any]:
        """
        Format output as image.

        Args:
        ----
            output: Output image data

        Returns:
        -------
            Formatted image data

        """
        if isinstance(output, dict) and ("url" in output or "data" in output):
            return output
        if isinstance(output, str) and (
            output.startswith(("http://", "https://", "data:")) or os.path.exists(output)
        ):
            return {"type": "url", "url": output}
        if isinstance(output, bytes):
            return {"type": "data", "data": output}
        msg = f"Unsupported image output type: {type(output)}"
        raise ValueError(msg)

    def to_message_content(self, data: dict[str, Any]) -> MessageContent:
        """
        Convert image data to MessageContent.

        Args:
        ----
            data: Image data (dict with 'url' or 'data')

        Returns:
        -------
            MessageContent object

        """
        if isinstance(data, dict):
            if "url" in data:
                return MessageContent(
                    type=ContentType.IMAGE,
                    text=None,
                    image_url=data["url"],
                    image_data=None,
                    audio_url=None,
                    audio_data=None,
                    video_url=None,
                    video_data=None,
                )
            if "data" in data:
                return MessageContent(
                    type=ContentType.IMAGE,
                    text=None,
                    image_url=None,
                    image_data=data["data"],
                    audio_url=None,
                    audio_data=None,
                    video_url=None,
                    video_data=None,
                )

        msg = f"Invalid image data format: {data}"
        raise ValueError(msg)

    @classmethod
    def from_message_content(cls, content: MessageContent) -> dict[str, Any]:
        """
        Convert MessageContent to image data.

        Args:
        ----
            content: MessageContent to convert

        Returns:
        -------
            Image data

        """
        if content.type != ContentType.IMAGE:
            msg = f"Expected IMAGE content type, got {content.type}"
            raise ValueError(msg)

        if content.image_url:
            return {"type": "url", "url": content.image_url}
        if content.image_data:
            return {"type": "data", "data": content.image_data}
        msg = "MessageContent has no image URL or data"
        raise ValueError(msg)
