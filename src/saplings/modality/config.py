from __future__ import annotations

"""
Configuration module for modality handling.

This module defines the configuration classes for modality handling.
"""


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    """Types of modalities supported by the system."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ModalityConfig(BaseModel):
    """Configuration for modality handling."""

    # Supported modalities
    supported_modalities: list[ModalityType] = Field(
        default_factory=lambda: [ModalityType.TEXT],
        description="List of supported modalities",
    )

    # Text modality settings
    text_max_length: int = Field(
        default=8192,
        description="Maximum text length in characters",
        gt=0,
    )

    # Image modality settings
    image_formats: list[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"],
        description="Supported image formats",
    )
    image_max_size_mb: float = Field(
        default=10.0,
        description="Maximum image size in MB",
        gt=0,
    )
    image_resize_dimensions: dict[str, int] | None = Field(
        default=None,
        description="Dimensions to resize images to (width, height)",
    )

    # Audio modality settings
    audio_formats: list[str] = Field(
        default_factory=lambda: ["mp3", "wav", "ogg", "m4a"],
        description="Supported audio formats",
    )
    audio_max_duration_seconds: int = Field(
        default=300,
        description="Maximum audio duration in seconds",
        gt=0,
    )
    audio_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
        gt=0,
    )

    # Video modality settings
    video_formats: list[str] = Field(
        default_factory=lambda: ["mp4", "mov", "avi", "webm"],
        description="Supported video formats",
    )
    video_max_duration_seconds: int = Field(
        default=60,
        description="Maximum video duration in seconds",
        gt=0,
    )
    video_max_size_mb: float = Field(
        default=100.0,
        description="Maximum video size in MB",
        gt=0,
    )

    # Handler settings
    custom_handlers: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom handlers for specific modalities",
    )

    # Conversion settings
    enable_auto_conversion: bool = Field(
        default=True,
        description="Whether to automatically convert between modalities when possible",
    )
    conversion_quality: int = Field(
        default=85,
        description="Quality for conversions (0-100)",
        ge=0,
        le=100,
    )

    # Cache settings
    enable_cache: bool = Field(
        default=True,
        description="Whether to cache processed modality data",
    )
    cache_dir: str = Field(
        default="./modality_cache",
        description="Directory for caching processed modality data",
    )
    cache_max_size_mb: int = Field(
        default=1024,
        description="Maximum cache size in MB",
        gt=0,
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    class Config:
        """Pydantic configuration."""

        extra = "ignore"
