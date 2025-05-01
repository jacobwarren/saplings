"""
Tests for the message module.

This module provides tests for the message classes in Saplings.
"""

import base64
import json

import pytest

from saplings.core.message import (
    ContentType,
    FunctionCall,
    FunctionDefinition,
    Message,
    MessageContent,
    MessageRole,
)


class TestMessageContent:
    """Test class for the MessageContent class."""

    def test_text_content(self):
        """Test creating text content."""
        content = MessageContent(type=ContentType.TEXT, text="Hello, world!")

        # Check properties
        assert content.type == ContentType.TEXT
        assert content.text == "Hello, world!"

        # Check dictionary representation
        content_dict = content.to_dict()
        assert content_dict["type"] == "text"
        assert content_dict["text"] == "Hello, world!"

    def test_image_content_with_url(self):
        """Test creating image content with a URL."""
        content = MessageContent(type=ContentType.IMAGE, image_url="https://example.com/image.jpg")

        # Check properties
        assert content.type == ContentType.IMAGE
        assert content.image_url == "https://example.com/image.jpg"

        # Check dictionary representation
        content_dict = content.to_dict()
        assert content_dict["type"] == "image_url"
        assert content_dict["image_url"]["url"] == "https://example.com/image.jpg"

    def test_image_content_with_data(self):
        """Test creating image content with data."""
        # Create a tiny 1x1 transparent PNG
        image_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )

        content = MessageContent(type=ContentType.IMAGE, image_data=image_data)

        # Check properties
        assert content.type == ContentType.IMAGE
        assert content.image_data == image_data

        # Check dictionary representation
        content_dict = content.to_dict()
        assert content_dict["type"] == "image_url"
        assert content_dict["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_audio_content_with_url(self):
        """Test creating audio content with a URL."""
        content = MessageContent(type=ContentType.AUDIO, audio_url="https://example.com/audio.mp3")

        # Check properties
        assert content.type == ContentType.AUDIO
        assert content.audio_url == "https://example.com/audio.mp3"

        # Check dictionary representation
        content_dict = content.to_dict()
        assert content_dict["type"] == "audio_url"
        assert content_dict["audio_url"]["url"] == "https://example.com/audio.mp3"

    def test_audio_content_with_data(self):
        """Test creating audio content with data."""
        # Create some dummy audio data
        audio_data = b"dummy audio data"

        content = MessageContent(type=ContentType.AUDIO, audio_data=audio_data)

        # Check properties
        assert content.type == ContentType.AUDIO
        assert content.audio_data == audio_data

        # Check dictionary representation
        content_dict = content.to_dict()
        assert content_dict["type"] == "audio_url"
        assert content_dict["audio_url"]["url"].startswith("data:audio/mp3;base64,")

    def test_video_content_with_url(self):
        """Test creating video content with a URL."""
        content = MessageContent(type=ContentType.VIDEO, video_url="https://example.com/video.mp4")

        # Check properties
        assert content.type == ContentType.VIDEO
        assert content.video_url == "https://example.com/video.mp4"

        # Check dictionary representation
        content_dict = content.to_dict()
        assert content_dict["type"] == "video_url"
        assert content_dict["video_url"]["url"] == "https://example.com/video.mp4"

    def test_video_content_with_data(self):
        """Test creating video content with data."""
        # Create some dummy video data
        video_data = b"dummy video data"

        content = MessageContent(type=ContentType.VIDEO, video_data=video_data)

        # Check properties
        assert content.type == ContentType.VIDEO
        assert content.video_data == video_data

        # Check dictionary representation
        content_dict = content.to_dict()
        assert content_dict["type"] == "video_url"
        assert content_dict["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_from_dict_text(self):
        """Test creating text content from a dictionary."""
        data = {"type": "text", "text": "Hello, world!"}
        content = MessageContent.from_dict(data)

        assert content.type == ContentType.TEXT
        assert content.text == "Hello, world!"

    def test_from_dict_image(self):
        """Test creating image content from a dictionary."""
        data = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        content = MessageContent.from_dict(data)

        assert content.type == ContentType.IMAGE
        assert content.image_url == "https://example.com/image.jpg"

    def test_from_dict_audio(self):
        """Test creating audio content from a dictionary."""
        data = {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}
        content = MessageContent.from_dict(data)

        assert content.type == ContentType.AUDIO
        assert content.audio_url == "https://example.com/audio.mp3"

    def test_from_dict_video(self):
        """Test creating video content from a dictionary."""
        data = {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        content = MessageContent.from_dict(data)

        assert content.type == ContentType.VIDEO
        assert content.video_url == "https://example.com/video.mp4"


class TestMessage:
    """Test class for the Message class."""

    def test_system_message(self):
        """Test creating a system message."""
        message = Message.system("You are a helpful assistant.")

        assert message.role == MessageRole.SYSTEM
        assert message.content == "You are a helpful assistant."

    def test_user_message(self):
        """Test creating a user message."""
        message = Message.user("Hello, how are you?")

        assert message.role == MessageRole.USER
        assert message.content == "Hello, how are you?"

    def test_assistant_message(self):
        """Test creating an assistant message."""
        message = Message.assistant("I'm doing well, thank you!")

        assert message.role == MessageRole.ASSISTANT
        assert message.content == "I'm doing well, thank you!"

    def test_function_message(self):
        """Test creating a function message."""
        message = Message.function("get_weather", '{"temperature": 22, "unit": "celsius"}')

        assert message.role == MessageRole.FUNCTION
        assert message.name == "get_weather"
        assert message.content == '{"temperature": 22, "unit": "celsius"}'

    def test_with_image(self):
        """Test creating a message with an image."""
        message = Message.with_image(
            text="What's in this image?", image_url="https://example.com/image.jpg"
        )

        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == ContentType.TEXT
        assert message.content[0].text == "What's in this image?"
        assert message.content[1].type == ContentType.IMAGE
        assert message.content[1].image_url == "https://example.com/image.jpg"

    def test_with_image_data(self):
        """Test creating a message with image data."""
        # Create a tiny 1x1 transparent PNG
        image_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )

        message = Message.with_image_data(text="What's in this image?", image_data=image_data)

        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == ContentType.TEXT
        assert message.content[0].text == "What's in this image?"
        assert message.content[1].type == ContentType.IMAGE
        assert message.content[1].image_data == image_data

    def test_with_audio(self):
        """Test creating a message with audio."""
        message = Message.with_audio(
            text="What's in this audio?", audio_url="https://example.com/audio.mp3"
        )

        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == ContentType.TEXT
        assert message.content[0].text == "What's in this audio?"
        assert message.content[1].type == ContentType.AUDIO
        assert message.content[1].audio_url == "https://example.com/audio.mp3"

    def test_with_audio_data(self):
        """Test creating a message with audio data."""
        # Create some dummy audio data
        audio_data = b"dummy audio data"

        message = Message.with_audio_data(text="What's in this audio?", audio_data=audio_data)

        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == ContentType.TEXT
        assert message.content[0].text == "What's in this audio?"
        assert message.content[1].type == ContentType.AUDIO
        assert message.content[1].audio_data == audio_data

    def test_with_video(self):
        """Test creating a message with video."""
        message = Message.with_video(
            text="What's in this video?", video_url="https://example.com/video.mp4"
        )

        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == ContentType.TEXT
        assert message.content[0].text == "What's in this video?"
        assert message.content[1].type == ContentType.VIDEO
        assert message.content[1].video_url == "https://example.com/video.mp4"

    def test_with_video_data(self):
        """Test creating a message with video data."""
        # Create some dummy video data
        video_data = b"dummy video data"

        message = Message.with_video_data(text="What's in this video?", video_data=video_data)

        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == ContentType.TEXT
        assert message.content[0].text == "What's in this video?"
        assert message.content[1].type == ContentType.VIDEO
        assert message.content[1].video_data == video_data

    def test_to_dict(self):
        """Test converting a message to a dictionary."""
        message = Message.user("Hello, world!")

        message_dict = message.to_dict()
        assert message_dict["role"] == "user"
        assert message_dict["content"] == "Hello, world!"

    def test_to_dict_with_function_call(self):
        """Test converting a message with a function call to a dictionary."""
        function_call = FunctionCall(name="get_weather", arguments={"location": "San Francisco"})

        message = Message.assistant(function_call=function_call)

        message_dict = message.to_dict()
        assert message_dict["role"] == "assistant"
        assert message_dict["function_call"]["name"] == "get_weather"
        assert message_dict["function_call"]["arguments"] == {"location": "San Francisco"}

    def test_from_dict(self):
        """Test creating a message from a dictionary."""
        data = {"role": "user", "content": "Hello, world!"}
        message = Message.from_dict(data)

        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"

    def test_from_dict_with_function_call(self):
        """Test creating a message with a function call from a dictionary."""
        data = {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "get_weather", "arguments": '{"location": "San Francisco"}'},
        }

        message = Message.from_dict(data)

        assert message.role == MessageRole.ASSISTANT
        assert message.function_call is not None
        assert message.function_call.name == "get_weather"
        assert message.function_call.arguments == {"location": "San Francisco"}


class TestFunctionDefinition:
    """Test class for the FunctionDefinition class."""

    def test_create_function_definition(self):
        """Test creating a function definition."""
        function_def = FunctionDefinition(
            name="get_weather",
            description="Get the weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature to use",
                },
            },
            required_parameters=["location"],
        )

        assert function_def.name == "get_weather"
        assert function_def.description == "Get the weather for a location"
        assert "location" in function_def.parameters
        assert "unit" in function_def.parameters
        assert function_def.required_parameters == ["location"]

    def test_to_dict(self):
        """Test converting a function definition to a dictionary."""
        function_def = FunctionDefinition(
            name="get_weather",
            description="Get the weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            required_parameters=["location"],
        )

        function_dict = function_def.to_dict()
        assert function_dict["name"] == "get_weather"
        assert function_dict["description"] == "Get the weather for a location"
        assert function_dict["parameters"]["type"] == "object"
        assert "location" in function_dict["parameters"]["properties"]
        assert function_dict["parameters"]["required"] == ["location"]

    def test_from_dict(self):
        """Test creating a function definition from a dictionary."""
        data = {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }

        function_def = FunctionDefinition.from_dict(data)

        assert function_def.name == "get_weather"
        assert function_def.description == "Get the weather for a location"
        assert "location" in function_def.parameters
        assert function_def.required_parameters == ["location"]


class TestFunctionCall:
    """Test class for the FunctionCall class."""

    def test_create_function_call(self):
        """Test creating a function call."""
        function_call = FunctionCall(
            name="get_weather",
            arguments={"location": "San Francisco", "unit": "celsius"},
            id="call_123",
        )

        assert function_call.name == "get_weather"
        assert function_call.arguments == {"location": "San Francisco", "unit": "celsius"}
        assert function_call.id == "call_123"

    def test_to_dict(self):
        """Test converting a function call to a dictionary."""
        function_call = FunctionCall(
            name="get_weather", arguments={"location": "San Francisco"}, id="call_123"
        )

        function_dict = function_call.to_dict()
        assert function_dict["name"] == "get_weather"
        assert function_dict["arguments"] == {"location": "San Francisco"}
        assert function_dict["id"] == "call_123"

    def test_from_dict(self):
        """Test creating a function call from a dictionary."""
        data = {
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}',
            "id": "call_123",
        }

        function_call = FunctionCall.from_dict(data)

        assert function_call.name == "get_weather"
        assert function_call.arguments == {"location": "San Francisco"}
        assert function_call.id == "call_123"

    def test_from_dict_with_json_string(self):
        """Test creating a function call from a dictionary with a JSON string."""
        data = {
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}',
        }

        function_call = FunctionCall.from_dict(data)

        assert function_call.name == "get_weather"
        assert function_call.arguments == {"location": "San Francisco"}
