from __future__ import annotations

"""
Execution service interface for Saplings.

This module defines the interface for execution operations that run prompts
with the model. This is a pure interface with no implementation details or
dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Import standard types
from saplings.core.lifecycle import ServiceState

# Forward references
Document = Any  # From saplings.memory.document


class IExecutionService(ABC):
    """Interface for execution operations."""

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the execution service.

        This method should be called after the service is created to ensure
        all dependencies are properly initialized.

        Raises
        ------
            InitializationError: If initialization fails

        """

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shut down the execution service.

        This method should be called when the service is no longer needed to
        ensure proper resource cleanup.

        Raises
        ------
            InitializationError: If shutdown fails

        """

    @property
    @abstractmethod
    def state(self) -> ServiceState:
        """
        Get the current state of the service.

        Returns
        -------
            ServiceState: Current state of the service

        """

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        documents: Optional[List[Document]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        validation_type: Optional[str] = None,
        verification_strategy: Optional[str] = None,  # For backward compatibility
    ) -> Any:
        """
        Execute a prompt with the model.

        Args:
        ----
            prompt: The prompt to execute
            documents: Optional documents to provide context
            functions: Optional function definitions
            function_call: Optional function call specification
            system_prompt: Optional system prompt
            temperature: Optional temperature for sampling
            max_tokens: Optional maximum number of tokens to generate
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds
            validation_type: Optional validation type to use
            verification_strategy: Optional verification strategy (for backward compatibility)

        Returns:
        -------
            Any: The execution result

        """

    @abstractmethod
    async def execute_with_verification(
        self,
        prompt: str,
        validation_type: str,
        documents: Optional[List[Document]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        trace_id: Optional[str] = None,
        verification_strategy: Optional[str] = None,  # For backward compatibility
    ) -> Any:
        """
        Execute a prompt with validation.

        Args:
        ----
            prompt: The prompt to execute
            validation_type: The validation type to use
            documents: Optional documents to provide context
            functions: Optional function definitions
            function_call: Optional function call specification
            trace_id: Optional trace ID for monitoring
            verification_strategy: Optional verification strategy (for backward compatibility)

        Returns:
        -------
            Any: The verified execution result

        """

    @abstractmethod
    async def execute_function(
        self, function_name: str, parameters: Dict[str, Any], trace_id: Optional[str] = None
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
