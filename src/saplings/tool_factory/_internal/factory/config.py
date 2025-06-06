from __future__ import annotations

"""
Configuration module for the tool factory system.

This module defines the configuration classes for the tool factory system.
"""


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SecurityLevel(str, Enum):
    """Security level for tool generation."""

    LOW = "low"  # Basic security checks
    MEDIUM = "medium"  # Standard security checks
    HIGH = "high"  # Strict security checks


class SandboxType(str, Enum):
    """Type of sandbox for tool execution."""

    NONE = "none"  # No sandboxing
    DOCKER = "docker"  # Docker-based sandbox
    E2B = "e2b"  # E2B-based sandbox


class SigningLevel(str, Enum):
    """Level of code signing for tools."""

    NONE = "none"  # No code signing
    BASIC = "basic"  # Basic code signing (hash verification)
    ADVANCED = "advanced"  # Advanced code signing (cryptographic signatures)


class ToolTemplate(BaseModel):
    """Configuration for a tool template."""

    id: str = Field(..., description="Unique identifier for the template")
    name: str = Field(..., description="Human-readable name for the template")
    description: str = Field(..., description="Description of the template's purpose")
    template_code: str = Field(..., description="Template code with placeholders")
    required_parameters: list[str] = Field(
        ..., description="List of required parameters for the template"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate the template ID."""
        if not v:
            msg = "Template ID cannot be empty"
            raise ValueError(msg)
        return v


class ToolSpecification(BaseModel):
    """Configuration for a tool specification."""

    id: str = Field(..., description="Unique identifier for the tool")
    name: str = Field(..., description="Human-readable name for the tool")
    description: str = Field(..., description="Description of the tool's purpose")
    template_id: str = Field(..., description="ID of the template to use")
    parameters: dict[str, Any] = Field(..., description="Parameters for the template")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("id", "template_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate the IDs."""
        if not v:
            msg = "ID cannot be empty"
            raise ValueError(msg)
        return v


class ToolFactoryConfig(BaseModel):
    """Configuration for the ToolFactory."""

    output_dir: str = Field(
        default="tools",
        description="Directory for generated tools",
    )
    security_level: SecurityLevel = Field(
        default=SecurityLevel.MEDIUM,
        description="Security level for tool generation",
    )
    enable_code_signing: bool = Field(
        default=False,
        description="Whether to enable code signing (deprecated, use signing_level instead)",
    )
    signing_level: SigningLevel = Field(
        default=SigningLevel.NONE,
        description="Level of code signing for tools",
    )
    signing_key_path: str | None = Field(
        default=None,
        description="Path to the signing key file (required for ADVANCED signing)",
    )
    sandbox_type: SandboxType = Field(
        default=SandboxType.NONE,
        description="Type of sandbox to use for tool execution",
    )
    docker_image: str | None = Field(
        default="python:3.9-slim",
        description="Docker image to use for sandboxed execution (only for DOCKER sandbox)",
    )
    e2b_api_key: str | None = Field(
        default=None,
        description="E2B API key for cloud sandbox (only for E2B sandbox)",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout in seconds for sandboxed execution",
    )

    # For backward compatibility
    sandbox_timeout: int | None = Field(
        default=None,
        description="Deprecated: Use timeout_seconds instead",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Validate the output directory."""
        if not v:
            msg = "Output directory cannot be empty"
            raise ValueError(msg)
        return v


class SecureHotLoaderConfig(BaseModel):
    """Configuration for the SecureHotLoader."""

    security_level: SecurityLevel = Field(
        default=SecurityLevel.HIGH,
        description="Security level for code loading",
    )
    signing_level: SigningLevel = Field(
        default=SigningLevel.BASIC,
        description="Level of code signing verification",
    )
    verification_key_path: str | None = Field(
        default=None,
        description="Path to the verification key file (required for ADVANCED signing)",
    )
    sandbox_type: SandboxType = Field(
        default=SandboxType.NONE,
        description="Type of sandbox to use for code execution",
    )
    docker_image: str | None = Field(
        default="python:3.9-slim",
        description="Docker image to use for sandboxed execution (only for DOCKER sandbox)",
    )
    e2b_api_key: str | None = Field(
        default=None,
        description="E2B API key for cloud sandbox (only for E2B sandbox)",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout in seconds for sandboxed execution",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
