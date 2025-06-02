from __future__ import annotations

"""
Memory Manager Builder module for Saplings.

This module provides a builder class for creating MemoryManager instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import Any

from saplings._internal.services.memory_manager import MemoryManager
from saplings.core._internal.builder import ServiceBuilder

logger = logging.getLogger(__name__)


class MemoryManagerBuilder(ServiceBuilder[MemoryManager]):
    """
    Builder for MemoryManager.

    This class provides a fluent interface for building MemoryManager instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for MemoryManager
    builder = MemoryManagerBuilder()

    # Configure the builder with dependencies and options
    manager = builder.with_memory_path("./agent_memory") \
                    .with_trace_manager(trace_manager) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the memory manager builder."""
        super().__init__(MemoryManager)
        self._memory_path = "./agent_memory"
        self._trace_manager = None

    def with_memory_path(self, memory_path: str) -> MemoryManagerBuilder:
        """
        Set the memory path.

        Args:
        ----
            memory_path: Path to store agent memory

        Returns:
        -------
            The builder instance for method chaining

        """
        self._memory_path = memory_path
        self.with_dependency("memory_path", memory_path)
        return self

    def with_trace_manager(self, trace_manager: Any) -> MemoryManagerBuilder:
        """
        Set the trace manager.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            The builder instance for method chaining

        """
        self._trace_manager = trace_manager
        self.with_dependency("trace_manager", trace_manager)
        return self

    def build(self) -> MemoryManager:
        """
        Build the memory manager instance with the configured dependencies.

        Returns
        -------
            The initialized memory manager instance

        """
        return super().build()
