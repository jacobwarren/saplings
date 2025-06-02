from __future__ import annotations

"""
Internal implementation of the patch module for self-healing capabilities.

This module provides classes for representing and managing patches.
"""


import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PatchStatus(str, Enum):
    """Status of a patch."""

    GENERATED = "generated"  # Patch has been generated
    APPLIED = "applied"  # Patch has been applied
    VALIDATED = "validated"  # Patch has been validated
    FAILED = "failed"  # Patch application failed
    INVALID = "invalid"  # Patch validation failed
    REVERTED = "reverted"  # Patch has been reverted


class Patch(BaseModel):
    """
    Representation of a code patch.

    A patch is a proposed fix for a code error, with metadata about its
    status, confidence, and other attributes.
    """

    patch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the patch",
    )
    description: str = Field(
        ...,
        description="Description of what the patch fixes",
    )
    code: str = Field(
        ...,
        description="The patched code",
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the patch (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    status: PatchStatus = Field(
        default=PatchStatus.GENERATED,
        description="Status of the patch",
    )
    error_type: str | None = Field(
        default=None,
        description="Type of error the patch fixes",
    )
    original_code: str | None = Field(
        default=None,
        description="The original code before patching",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    created_at: str | None = Field(
        default=None,
        description="Timestamp when the patch was created",
    )
    applied_at: str | None = Field(
        default=None,
        description="Timestamp when the patch was applied",
    )
    validated_at: str | None = Field(
        default=None,
        description="Timestamp when the patch was validated",
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the patch to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        return {
            "patch_id": self.patch_id,
            "description": self.description,
            "code": self.code,
            "confidence": self.confidence,
            "status": self.status,
            "error_type": self.error_type,
            "original_code": self.original_code,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
            "validated_at": self.validated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Patch":
        """
        Create a patch from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            Patch: Patch instance

        """
        return cls(**data)
