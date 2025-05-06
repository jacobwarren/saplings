from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from saplings.core.interfaces import IModalityService
from saplings.modality.config import ModalityConfig, ModalityType
from saplings.services.modality_service import ModalityService

"""
Unit tests for the modality service.
"""


class TestModalityService:
    """Test the modality service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create modality service
        self.config = ModalityConfig(
            supported_modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO]
        )
        self.service = ModalityService(config=self.config)

    def test_initialization(self) -> None:
        """Test modality service initialization."""
        assert self.service.config is self.config
        assert "text" in self.service.handlers
        assert "image" in self.service.handlers
        assert "audio" in self.service.handlers

    def test_process_text(self) -> None:
        """Test processing text modality."""
        # Process text
        result = self.service.process("text", "This is a test text.")

        # Verify result
        assert result == "This is a test text."

    def test_process_image_url(self) -> None:
        """Test processing image modality with URL."""
        # Mock the image handler
        mock_handler = MagicMock()
        mock_handler.process.return_value = "Image description: A test image"
        self.service.handlers["image"] = mock_handler

        # Process image URL
        result = self.service.process("image", "https://example.com/test.jpg")

        # Verify result
        assert result == "Image description: A test image"
        mock_handler.process.assert_called_once_with("https://example.com/test.jpg")

    def test_process_audio(self) -> None:
        """Test processing audio modality."""
        # Mock the audio handler
        mock_handler = MagicMock()
        mock_handler.process.return_value = "Transcription: This is a test audio"
        self.service.handlers["audio"] = mock_handler

        # Process audio
        result = self.service.process("audio", "test_audio.mp3")

        # Verify result
        assert result == "Transcription: This is a test audio"
        mock_handler.process.assert_called_once_with("test_audio.mp3")

    def test_process_unsupported_modality(self) -> None:
        """Test processing unsupported modality."""
        # Process unsupported modality
        with pytest.raises(ValueError):
            self.service.process("video", "test_video.mp4")

    def test_register_handler(self) -> None:
        """Test registering a custom handler."""
        # Create a custom handler
        mock_handler = MagicMock()
        mock_handler.process.return_value = "Custom handler result"

        # Register the handler
        self.service.register_handler("custom", mock_handler)

        # Process with custom handler
        result = self.service.process("custom", "custom_input")

        # Verify result
        assert result == "Custom handler result"
        mock_handler.process.assert_called_once_with("custom_input")

    def test_get_supported_modalities(self) -> None:
        """Test getting supported modalities."""
        # Get supported modalities
        modalities = self.service.get_supported_modalities()

        # Verify modalities
        assert "text" in modalities
        assert "image" in modalities
        assert "audio" in modalities

    def test_interface_compliance(self) -> None:
        """Test that ModalityService implements IModalityService."""
        assert isinstance(self.service, IModalityService)

        # Check required methods
        assert hasattr(self.service, "process")
        assert hasattr(self.service, "register_handler")
        assert hasattr(self.service, "get_supported_modalities")
