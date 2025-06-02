from __future__ import annotations

"""
Self-Healing Service Builder module for Saplings.

This module provides a builder class for creating SelfHealingService instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import Any

from saplings._internal.services.self_healing_service import SelfHealingService
from saplings.core._internal.builder import ServiceBuilder

logger = logging.getLogger(__name__)


class SelfHealingServiceBuilder(ServiceBuilder[SelfHealingService]):
    """
    Builder for SelfHealingService.

    This class provides a fluent interface for building SelfHealingService instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for SelfHealingService
    builder = SelfHealingServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_patch_generator(patch_generator) \
                    .with_success_pair_collector(success_pair_collector) \
                    .with_enabled(True) \
                    .with_trace_manager(trace_manager) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the self-healing service builder."""
        super().__init__(SelfHealingService)
        self._patch_generator = None
        self._success_pair_collector = None
        self._enabled = True
        self._trace_manager = None
        self.require_dependency("patch_generator")
        self.require_dependency("success_pair_collector")

    def with_patch_generator(self, patch_generator: Any) -> SelfHealingServiceBuilder:
        """
        Set the patch generator.

        Args:
        ----
            patch_generator: Patch generator for self-healing

        Returns:
        -------
            The builder instance for method chaining

        """
        self._patch_generator = patch_generator
        self.with_dependency("patch_generator", patch_generator)
        return self

    def with_success_pair_collector(self, collector: Any) -> SelfHealingServiceBuilder:
        """
        Set the success pair collector.

        Args:
        ----
            collector: Success pair collector for self-healing

        Returns:
        -------
            The builder instance for method chaining

        """
        self._success_pair_collector = collector
        self.with_dependency("success_pair_collector", collector)
        return self

    def with_enabled(self, enabled: bool) -> SelfHealingServiceBuilder:
        """
        Set whether self-healing is enabled.

        Args:
        ----
            enabled: Whether self-healing is enabled

        Returns:
        -------
            The builder instance for method chaining

        """
        self._enabled = enabled
        self.with_dependency("enabled", enabled)
        return self

    def with_trace_manager(self, trace_manager: Any) -> SelfHealingServiceBuilder:
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

    def build(self) -> SelfHealingService:
        """
        Build the self-healing service instance with the configured dependencies.

        Returns
        -------
            The initialized self-healing service instance

        """
        return super().build()
