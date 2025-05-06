from __future__ import annotations

"""
Execution service interface for Saplings.

This module defines the interface for execution operations that run prompts
with the model and handle verification. This is a pure interface with no
implementation details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any

# Forward references
Document = Any  # From saplings.memory.document


class IExecutionService(ABC):
    """Interface for execution operations."""

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        documents: list[Document] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Any:
        """
        Execute a prompt with the model.

        Args:
        ----
            prompt: The prompt to execute
            documents: Optional documents to provide context
            functions: Optional function definitions
            function_call: Optional function call specification
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Any: The execution result

        """

    @abstractmethod
    async def execute_with_verification(
        self,
        prompt: str,
        verification_strategy: str,
        documents: list[Document] | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Any:
        """
        Execute a prompt with verification.

        Args:
        ----
            prompt: The prompt to execute
            verification_strategy: The verification strategy to use
            documents: Optional documents to provide context
            functions: Optional function definitions
            function_call: Optional function call specification
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Any: The verified execution result

        """

    @abstractmethod
    async def execute_function(
        self, function_name: str, parameters: dict[str, Any], trace_id: str | None = None
    ) -> Any:
        """
        Execute a specific function.

        Args:
        ----
            function_name: The name of the function to execute
            parameters: Function parameters
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Any: The function result

        """

    @property
    @abstractmethod
    def executor(self):
        """
        Get the underlying executor.

        Returns
        -------
            Any: The executor instance

        """
