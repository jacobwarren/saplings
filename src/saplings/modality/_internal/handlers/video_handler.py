from __future__ import annotations

"""
Video modality handler for Saplings.

This module provides the handler implementation for video modality.
"""

import os
from typing import Any

from saplings.core.message import ContentType, MessageContent
from saplings.modality._internal.handlers.modality_handler import ModalityHandler


class VideoHandler(ModalityHandler):
    """Handler for video modality."""

    async def process_input(self, input_data: Any) -> dict[str, Any]:
        """
        Process video input.

        Args:
        ----
            input_data: Input video data (URL, file path, or bytes)

        Returns:
        -------
            Processed video data

        """
        if isinstance(input_data, str):
            # Check if it's a URL or file path
            if input_data.startswith(("http://", "https://", "data:")):
                return {"type": "url", "url": input_data}
            if os.path.exists(input_data):
                # Read file content
                with open(input_data, "rb") as f:
                    video_data = f.read()
                return {"type": "data", "data": video_data}
            msg = f"Invalid video path or URL: {input_data}"
            raise ValueError(msg)
        if isinstance(input_data, bytes):
            return {"type": "data", "data": input_data}
        if isinstance(input_data, dict) and ("url" in input_data or "data" in input_data):
            return input_data
        msg = f"Unsupported video input type: {type(input_data)}"
        raise ValueError(msg)

    async def format_output(self, output: Any) -> dict[str, Any]:
        """
        Format output as video.

        Args:
        ----
            output: Output video data

        Returns:
        -------
            Formatted video data

        """
        if isinstance(output, dict) and ("url" in output or "data" in output):
            return output
        if isinstance(output, str) and (
            output.startswith(("http://", "https://", "data:")) or os.path.exists(output)
        ):
            return {"type": "url", "url": output}
        if isinstance(output, bytes):
            return {"type": "data", "data": output}
        msg = f"Unsupported video output type: {type(output)}"
        raise ValueError(msg)

    def to_message_content(self, data: dict[str, Any]) -> MessageContent:
        """
        Convert video data to MessageContent.

        Args:
        ----
            data: Video data (dict with 'url' or 'data')

        Returns:
        -------
            MessageContent object

        """
        if isinstance(data, dict):
            if "url" in data:
                return MessageContent(
                    type=ContentType.VIDEO,
                    text=None,
                    image_url=None,
                    image_data=None,
                    audio_url=None,
                    audio_data=None,
                    video_url=data["url"],
                    video_data=None,
                )
            if "data" in data:
                return MessageContent(
                    type=ContentType.VIDEO,
                    text=None,
                    image_url=None,
                    image_data=None,
                    audio_url=None,
                    audio_data=None,
                    video_url=None,
                    video_data=data["data"],
                )

        msg = f"Invalid video data format: {data}"
        raise ValueError(msg)

    @classmethod
    def from_message_content(cls, content: MessageContent) -> dict[str, Any]:
        """
        Convert MessageContent to video data.

        Args:
        ----
            content: MessageContent to convert

        Returns:
        -------
            Video data

        """
        if content.type != ContentType.VIDEO:
            msg = f"Expected VIDEO content type, got {content.type}"
            raise ValueError(msg)

        if content.video_url:
            return {"type": "url", "url": content.video_url}
        if content.video_data:
            return {"type": "data", "data": content.video_data}
        msg = "MessageContent has no video URL or data"
        raise ValueError(msg)
