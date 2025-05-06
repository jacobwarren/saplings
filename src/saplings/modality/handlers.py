from __future__ import annotations

"""
Modality handlers for Saplings.

This module provides handlers for different modalities (text, image, audio, video)
that can be used by agents to process and generate content in different formats.
"""


import logging
import os
from typing import TYPE_CHECKING, Any

from saplings.core.message import ContentType, MessageContent

if TYPE_CHECKING:
    from saplings.core.model_adapter import LLM

logger = logging.getLogger(__name__)


class ModalityHandler:
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


def get_handler_for_modality(modality: str, model: LLM) -> ModalityHandler:
    """
    Get the appropriate handler for a modality.

    Args:
    ----
        modality: Modality name (text, image, audio, video)
        model: LLM model to use for processing

    Returns:
    -------
        ModalityHandler for the specified modality

    """
    handlers = {
        "text": TextHandler,
        "image": ImageHandler,
        "audio": AudioHandler,
        "video": VideoHandler,
    }

    if modality not in handlers:
        msg = f"Unsupported modality: {modality}"
        raise ValueError(msg)

    return handlers[modality](model)
