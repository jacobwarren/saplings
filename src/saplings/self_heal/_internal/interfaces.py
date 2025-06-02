from __future__ import annotations

"""
Internal interfaces for self-healing components.

This module defines interfaces for self-healing components to ensure proper
separation of concerns and decouple implementations from usage patterns.
"""


from abc import ABC, abstractmethod
from typing import Any


class IPatchGenerator(ABC):
    """Interface for generating patches to fix code errors."""

    @abstractmethod
    async def generate_patch(
        self,
        error_message: str,
        code_context: str,
    ) -> dict[str, Any]:
        """
        Generate a patch for a failed execution.

        Args:
        ----
            error_message: Error message from failed execution
            code_context: Code context where error occurred

        Returns:
        -------
            Dictionary with patch information including success status and patch details

        """

    @abstractmethod
    def validate_patch(self, patched_code: str) -> tuple[bool, str | None]:
        """
        Validate a patched code by analyzing and/or executing it.

        Args:
        ----
            patched_code: Patched code to validate

        Returns:
        -------
            Tuple[bool, Optional[str]]: (is_valid, error_message)

        """


class ISuccessPairCollector(ABC):
    """Interface for collecting and managing success pairs for future learning."""

    @abstractmethod
    async def collect(
        self,
        input_text: str,
        output_text: str,
        context: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Collect a success pair.

        Args:
        ----
            input_text: Input text (e.g., prompt, error, etc.)
            output_text: Output text (e.g., response, fix, etc.)
            context: Context documents
            metadata: Additional metadata

        """

    @abstractmethod
    async def get_all_pairs(self):
        """
        Get all collected pairs.

        Returns
        -------
            List[Dict[str, Any]]: List of collected pairs

        """

    @abstractmethod
    def clear(self):
        """Clear all collected pairs."""

    @abstractmethod
    def get_statistics(self):
        """
        Get statistics about the collected pairs.

        Returns
        -------
            Dict: Statistics about the collected pairs

        """


class IAdapterManager(ABC):
    """Interface for managing model adapters for self-healing."""

    @abstractmethod
    async def train_adapter(
        self,
        pairs: list[dict[str, Any]],
        adapter_name: str,
    ) -> dict[str, Any]:
        """
        Train an adapter using success pairs.

        Args:
        ----
            pairs: List of success pairs for training
            adapter_name: Name for the trained adapter

        Returns:
        -------
            Training results

        """

    @abstractmethod
    async def list_adapters(self):
        """
        List all available adapters.

        Returns
        -------
            List of adapter names

        """

    @abstractmethod
    async def load_adapter(self, adapter_name: str) -> bool:
        """
        Load an adapter.

        Args:
        ----
            adapter_name: Name of the adapter to load

        Returns:
        -------
            Whether the adapter was successfully loaded

        """

    @abstractmethod
    async def unload_adapter(self):
        """
        Unload the current adapter.

        Returns
        -------
            Whether the adapter was successfully unloaded

        """
