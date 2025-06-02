from __future__ import annotations

"""
Message module for Saplings.

This module defines the message classes used for communication with LLMs.
"""


import base64
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles for messages in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ContentType(str, Enum):
    """Types of content in a message."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class FunctionDefinition(BaseModel):
    """Definition of a function that can be called by a model."""

    name: str = Field(..., description="Name of the function")
    description: str = Field(..., description="Description of the function")
    parameters: dict[str, Any] = Field(..., description="Parameters of the function")
    required_parameters: list[str] = Field(
        default_factory=list, description="List of required parameter names"
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the function definition to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the function

        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_parameters,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionDefinition":
        """
        Create a function definition from a dictionary.

        Args:
        ----
            data: Dictionary representation of the function

        Returns:
        -------
            FunctionDefinition: The function definition

        """
        parameters = data.get("parameters", {}).get("properties", {})
        required = data.get("parameters", {}).get("required", [])

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=parameters,
            required_parameters=required,
        )


class FunctionCall(BaseModel):
    """A function call made by a model."""

    name: str = Field(..., description="Name of the function to call")
    arguments: dict[str, Any] = Field(..., description="Arguments to pass to the function")
    id: str | None = Field(None, description="Unique identifier for the function call")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the function call to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the function call

        """
        result = {
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.id:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionCall":
        """
        Create a function call from a dictionary.

        Args:
        ----
            data: Dictionary representation of the function call

        Returns:
        -------
            FunctionCall: The function call

        """
        # Handle arguments that might be a JSON string
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}  # fallback to empty dict if not valid JSON
        if not isinstance(arguments, dict):
            arguments = {}

        return cls(
            name=data.get("name", ""),
            arguments=arguments,
            id=data.get("id"),
        )


class MessageContent(BaseModel):
    """Content of a message."""

    type: ContentType = Field(..., description="Type of content")
    text: str | None = Field(None, description="Text content (for TEXT type)")
    image_url: str | None = Field(None, description="URL of an image (for IMAGE type)")
    image_data: bytes | None = Field(None, description="Raw image data (for IMAGE type)")
    audio_url: str | None = Field(None, description="URL of an audio file (for AUDIO type)")
    audio_data: bytes | None = Field(None, description="Raw audio data (for AUDIO type)")
    video_url: str | None = Field(None, description="URL of a video file (for VIDEO type)")
    video_data: bytes | None = Field(None, description="Raw video data (for VIDEO type)")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the message content to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the content

        """
        if self.type == ContentType.TEXT:
            return {
                "type": "text",
                "text": self.text,
            }
        if self.type == ContentType.IMAGE:
            if self.image_url:
                return {
                    "type": "image_url",
                    "image_url": {"url": self.image_url},
                }
            if self.image_data:
                # Base64 encode the image data
                b64_data = base64.b64encode(self.image_data).decode("utf-8")
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                }
        elif self.type == ContentType.AUDIO:
            if self.audio_url:
                return {
                    "type": "audio_url",
                    "audio_url": {"url": self.audio_url},
                }
            if self.audio_data:
                # Base64 encode the audio data
                b64_data = base64.b64encode(self.audio_data).decode("utf-8")
                return {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/mp3;base64,{b64_data}"},
                }
        elif self.type == ContentType.VIDEO:
            if self.video_url:
                return {
                    "type": "video_url",
                    "video_url": {"url": self.video_url},
                }
            if self.video_data:
                # Base64 encode the video data
                b64_data = base64.b64encode(self.video_data).decode("utf-8")
                return {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{b64_data}"},
                }

        return {"type": self.type.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageContent":
        """
        Create message content from a dictionary.

        Args:
        ----
            data: Dictionary representation of the content

        Returns:
        -------
            MessageContent: The message content

        """
        content_type = data.get("type", "text")

        if content_type == "text":
            return cls(
                type=ContentType.TEXT,
                text=data.get("text", ""),
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            )
        if content_type == "image_url":
            image_url = data.get("image_url", {}).get("url", "")
            return cls(
                type=ContentType.IMAGE,
                text=None,
                image_url=image_url,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            )
        if content_type == "image":
            # This is for backward compatibility
            return cls(
                type=ContentType.IMAGE,
                text=None,
                image_url=data.get("url", ""),
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            )
        if content_type == "audio_url":
            audio_url = data.get("audio_url", {}).get("url", "")
            return cls(
                type=ContentType.AUDIO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=audio_url,
                audio_data=None,
                video_url=None,
                video_data=None,
            )
        if content_type == "audio":
            # This is for backward compatibility
            return cls(
                type=ContentType.AUDIO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=data.get("url", ""),
                audio_data=None,
                video_url=None,
                video_data=None,
            )
        if content_type == "video_url":
            video_url = data.get("video_url", {}).get("url", "")
            return cls(
                type=ContentType.VIDEO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=video_url,
                video_data=None,
            )
        if content_type == "video":
            # This is for backward compatibility
            return cls(
                type=ContentType.VIDEO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=data.get("url", ""),
                video_data=None,
            )

        # Default to text if unknown type
        return cls(
            type=ContentType.TEXT,
            text=str(data),
            image_url=None,
            image_data=None,
            audio_url=None,
            audio_data=None,
            video_url=None,
            video_data=None,
        )


class Message(BaseModel):
    """A message in a conversation."""

    role: MessageRole = Field(..., description="Role of the message sender")
    content: str | list[MessageContent] | None = Field(None, description="Content of the message")
    name: str | None = Field(None, description="Name of the function (for FUNCTION role)")
    function_call: FunctionCall | None = Field(
        None, description="Function call (for ASSISTANT role)"
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the message to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of the message

        """
        result = {"role": self.role.value}

        # Handle content
        if isinstance(self.content, str):
            result["content"] = self.content or ""
        elif (
            isinstance(self.content, list)
            and len(self.content) == 1
            and self.content[0].type == ContentType.TEXT
        ):
            # Simple text content
            result["content"] = self.content[0].text or ""
        elif isinstance(self.content, list) and self.content:
            # Complex content
            result["content"] = json.dumps([c.to_dict() for c in self.content])

        # Add name for function messages
        if self.name:
            result["name"] = self.name

        # Add function call for assistant messages
        if self.function_call:
            result["function_call"] = json.dumps(self.function_call.to_dict())

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """
        Create a message from a dictionary.

        Args:
        ----
            data: Dictionary representation of the message

        Returns:
        -------
            Message: The message

        """
        role = data.get("role", "user")
        content = data.get("content")
        name = data.get("name")
        function_call_data = data.get("function_call")

        # Handle content
        if isinstance(content, str):
            processed_content = content
        elif isinstance(content, list):
            processed_content = [MessageContent.from_dict(c) for c in content]
        else:
            processed_content = None

        # Handle function call
        function_call = None
        if function_call_data:
            function_call = FunctionCall.from_dict(function_call_data)

        return cls(
            role=role,
            content=processed_content,
            name=name,
            function_call=function_call,
        )

    @classmethod
    def system(cls, content: str) -> "Message":
        """
        Create a system message.

        Args:
        ----
            content: Content of the message

        Returns:
        -------
            Message: The system message

        """
        return cls(role=MessageRole.SYSTEM, content=content, name=None, function_call=None)

    @classmethod
    def user(cls, content: str | list[MessageContent]) -> "Message":
        """
        Create a user message.

        Args:
        ----
            content: Content of the message

        Returns:
        -------
            Message: The user message

        """
        return cls(role=MessageRole.USER, content=content, name=None, function_call=None)

    @classmethod
    def assistant(
        cls, content: str | None = None, function_call: FunctionCall | None = None
    ) -> "Message":
        """
        Create an assistant message.

        Args:
        ----
            content: Content of the message
            function_call: Function call

        Returns:
        -------
            Message: The assistant message

        """
        return cls(
            role=MessageRole.ASSISTANT, content=content, name=None, function_call=function_call
        )

    @classmethod
    def function(cls, name: str, content: str) -> "Message":
        """
        Create a function message.

        Args:
        ----
            name: Name of the function
            content: Content of the message (function result)

        Returns:
        -------
            Message: The function message

        """
        return cls(role=MessageRole.FUNCTION, content=content, name=name, function_call=None)

    @classmethod
    def tool(cls, name: str, content: str) -> "Message":
        """
        Create a tool message.

        Args:
        ----
            name: Name of the tool
            content: Content of the message (tool result)

        Returns:
        -------
            Message: The tool message

        """
        return cls(role=MessageRole.TOOL, content=content, name=name, function_call=None)

    @classmethod
    def with_image(cls, text: str, image_url: str) -> "Message":
        """
        Create a user message with text and an image.

        Args:
        ----
            text: Text content
            image_url: URL of the image

        Returns:
        -------
            Message: The user message with text and image

        """
        content = [
            MessageContent(
                type=ContentType.TEXT,
                text=text,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
            MessageContent(
                type=ContentType.IMAGE,
                text=None,
                image_url=image_url,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
        ]
        return cls(role=MessageRole.USER, content=content, name=None, function_call=None)

    @classmethod
    def with_image_data(cls, text: str, image_data: bytes) -> "Message":
        """
        Create a user message with text and image data.

        Args:
        ----
            text: Text content
            image_data: Raw image data

        Returns:
        -------
            Message: The user message with text and image

        """
        content = [
            MessageContent(
                type=ContentType.TEXT,
                text=text,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
            MessageContent(
                type=ContentType.IMAGE,
                text=None,
                image_url=None,
                image_data=image_data,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
        ]
        return cls(role=MessageRole.USER, content=content, name=None, function_call=None)

    @classmethod
    def with_audio(cls, text: str, audio_url: str) -> "Message":
        """
        Create a user message with text and an audio file.

        Args:
        ----
            text: Text content
            audio_url: URL of the audio file

        Returns:
        -------
            Message: The user message with text and audio

        """
        content = [
            MessageContent(
                type=ContentType.TEXT,
                text=text,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
            MessageContent(
                type=ContentType.AUDIO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=audio_url,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
        ]
        return cls(role=MessageRole.USER, content=content, name=None, function_call=None)

    @classmethod
    def with_audio_data(cls, text: str, audio_data: bytes) -> "Message":
        """
        Create a user message with text and audio data.

        Args:
        ----
            text: Text content
            audio_data: Raw audio data

        Returns:
        -------
            Message: The user message with text and audio

        """
        content = [
            MessageContent(
                type=ContentType.TEXT,
                text=text,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
            MessageContent(
                type=ContentType.AUDIO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=audio_data,
                video_url=None,
                video_data=None,
            ),
        ]
        return cls(role=MessageRole.USER, content=content, name=None, function_call=None)

    @classmethod
    def with_video(cls, text: str, video_url: str) -> "Message":
        """
        Create a user message with text and a video file.

        Args:
        ----
            text: Text content
            video_url: URL of the video file

        Returns:
        -------
            Message: The user message with text and video

        """
        content = [
            MessageContent(
                type=ContentType.TEXT,
                text=text,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
            MessageContent(
                type=ContentType.VIDEO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=video_url,
                video_data=None,
            ),
        ]
        return cls(role=MessageRole.USER, content=content, name=None, function_call=None)

    @classmethod
    def with_video_data(cls, text: str, video_data: bytes) -> "Message":
        """
        Create a user message with text and video data.

        Args:
        ----
            text: Text content
            video_data: Raw video data

        Returns:
        -------
            Message: The user message with text and video

        """
        content = [
            MessageContent(
                type=ContentType.TEXT,
                text=text,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=None,
            ),
            MessageContent(
                type=ContentType.VIDEO,
                text=None,
                image_url=None,
                image_data=None,
                audio_url=None,
                audio_data=None,
                video_url=None,
                video_data=video_data,
            ),
        ]
        return cls(role=MessageRole.USER, content=content, name=None, function_call=None)
