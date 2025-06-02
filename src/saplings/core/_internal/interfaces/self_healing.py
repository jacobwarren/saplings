from __future__ import annotations

"""
Self-healing service interface for Saplings.

This module defines the interface for self-healing operations that collect
success pairs and generate patches. This is a pure interface with no
implementation details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SelfHealingConfig:
    """Configuration for self-healing operations."""

    enabled: bool = True
    max_success_pairs: int = 100
    max_patches: int = 50
    auto_apply_patches: bool = False
    model_name: Optional[str] = None


@dataclass
class SelfHealingResult:
    """Result of a self-healing operation."""

    success: bool
    patch_id: Optional[str] = None
    patch_content: Optional[str] = None
    applied: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ISelfHealingService(ABC):
    """Interface for self-healing operations."""

    @abstractmethod
    async def collect_success_pair(
        self,
        input_text: str,
        output_text: str,
        context: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Collect a success pair for future improvements.

        Args:
        ----
            input_text: Input text that produced the successful output
            output_text: Successful output text
            context: Optional contextual information
            metadata: Optional metadata
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: Information about the collected success pair

        """

    @abstractmethod
    async def get_all_success_pairs(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """
        Get all collected success pairs.

        Args:
        ----
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            List[Dict[str, Any]]: All success pairs

        """

    @abstractmethod
    async def generate_patch(
        self,
        failure_input: str,
        failure_output: str,
        desired_output: str | None = None,
        context: list[str] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a patch for a failed execution.

        Args:
        ----
            failure_input: Input that resulted in failure
            failure_output: Failed output
            desired_output: Optional desired output
            context: Optional contextual information
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: The generated patch

        """

    @abstractmethod
    async def apply_patch(self, patch_id: str, trace_id: str | None = None) -> bool:
        """
        Apply a patch.

        Args:
        ----
            patch_id: ID of the patch to apply
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            bool: Whether the patch was successfully applied

        """

    @property
    @abstractmethod
    def enabled(self):
        """
        Whether self-healing is enabled.

        Returns
        -------
            bool: Enabled status

        """

    @abstractmethod
    async def heal(
        self,
        failure_input: str,
        failure_output: str,
        config: Optional[SelfHealingConfig] = None,
        trace_id: Optional[str] = None,
    ) -> SelfHealingResult:
        """
        Attempt to heal a failure by generating and applying a patch.

        Args:
        ----
            failure_input: Input that resulted in failure
            failure_output: Failed output
            config: Optional self-healing configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            SelfHealingResult: Result of the healing operation

        """
