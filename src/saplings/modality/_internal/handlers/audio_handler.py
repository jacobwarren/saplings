from __future__ import annotations

"""
Audio modality handler for Saplings.

This module provides the handler implementation for audio modality.
"""

import os
from typing import Any

from saplings.core.message import ContentType, MessageContent
from saplings.modality._internal.handlers.modality_handler import ModalityHandler


class AudioHandler(ModalityHandler):
    """Handler for audio modality."""

    async def process_input(self, input_data: Any) -> dict[str, Any]:
        """
        Process audio input.

        Args:
        ----
            input_data: Input audio data (URL, file path, or bytes)

        Returns:
        -------
            Processed audio data

        """
        if isinstance(input_data, str):
            # Check if it's a URL or file path
            if input_data.startswith(("http://", "https://", "data:")):
                return {"type": "url", "url": input_data}
            if os.path.exists(input_data):
                # Read file content
                with open(input_data, "rb") as f:
                    audio_data = f.read()
                return {"type": "data", "data": audio_data}
            msg = f"Invalid audio path or URL: {input_data}"
            raise ValueError(msg)
        if isinstance(input_data, bytes):
            return {"type": "data", "data": input_data}
        if isinstance(input_data, dict) and ("url" in input_data or "data" in input_data):
            return input_data
        msg = f"Unsupported audio input type: {type(input_data)}"
        raise ValueError(msg)

    async def format_output(self, output: Any) -> dict[str, Any]:
        """
        Format output as audio.

        Args:
        ----
            output: Output audio data

        Returns:
        -------
            Formatted audio data

        """
        if isinstance(output, dict) and ("url" in output or "data" in output):
            return output
        if isinstance(output, str) and (
            output.startswith(("http://", "https://", "data:")) or os.path.exists(output)
        ):
            return {"type": "url", "url": output}
        if isinstance(output, bytes):
            return {"type": "data", "data": output}
        msg = f"Unsupported audio output type: {type(output)}"
        raise ValueError(msg)

    def to_message_content(self, data: dict[str, Any]) -> MessageContent:
        """
        Convert audio data to MessageContent.

        Args:
        ----
            data: Audio data (dict with 'url' or 'data')

        Returns:
        -------
            MessageContent object

        """
        if isinstance(data, dict):
            if "url" in data:
                return MessageContent(
                    type=ContentType.AUDIO,
                    text=None,
                    image_url=None,
                    image_data=None,
                    audio_url=data["url"],
                    audio_data=None,
                    video_url=None,
                    video_data=None,
                )
            if "data" in data:
                return MessageContent(
                    type=ContentType.AUDIO,
                    text=None,
                    image_url=None,
                    image_data=None,
                    audio_url=None,
                    audio_data=data["data"],
                    video_url=None,
                    video_data=None,
                )

        msg = f"Invalid audio data format: {data}"
        raise ValueError(msg)

    @classmethod
    def from_message_content(cls, content: MessageContent) -> dict[str, Any]:
        """
        Convert MessageContent to audio data.

        Args:
        ----
            content: MessageContent to convert

        Returns:
        -------
            Audio data

        """
        if content.type != ContentType.AUDIO:
            msg = f"Expected AUDIO content type, got {content.type}"
            raise ValueError(msg)

        if content.audio_url:
            return {"type": "url", "url": content.audio_url}
        if content.audio_data:
            return {"type": "data", "data": content.audio_data}
        msg = "MessageContent has no audio URL or data"
        raise ValueError(msg)
