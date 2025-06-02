from __future__ import annotations

"""
Planner Service Builder API module for Saplings.

This module provides the planner service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class PlannerServiceBuilder:
    """
    Builder for creating PlannerService instances with a fluent interface.

    This builder provides a convenient way to configure and create PlannerService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the planner service builder."""
        # Import here to avoid circular imports
        self._model = None
        self._budget = 0.0
        self._config = {}

    def with_model(self, model: Any) -> "PlannerServiceBuilder":
        """
        Set the model for the planner service.

        Args:
        ----
            model: Model to use for planning

        Returns:
        -------
            Self for method chaining

        """
        self._model = model
        return self

    def with_budget(self, budget: float) -> "PlannerServiceBuilder":
        """
        Set the budget for the planner service.

        Args:
        ----
            budget: Budget for planning in USD

        Returns:
        -------
            Self for method chaining

        """
        self._budget = budget
        return self

    def with_trace_manager(self, trace_manager: Any) -> "PlannerServiceBuilder":
        """
        Set the trace manager for the planner service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._config["trace_manager"] = trace_manager
        return self

    def with_config(self, **kwargs) -> "PlannerServiceBuilder":
        """
        Set additional configuration options for the planner service.

        Args:
        ----
            **kwargs: Additional configuration options

        Returns:
        -------
            Self for method chaining

        """
        self._config.update(kwargs)
        return self

    def build(self) -> Any:
        """
        Build the planner service instance.

        Returns
        -------
            PlannerService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import the PlannerConfig
            from saplings.api.core import PlannerConfig
            from saplings.api.services.planner import PlannerService

            # Create a config object
            config = PlannerConfig(
                budget=self._budget,
                **{k: v for k, v in self._config.items() if k not in ["trace_manager"]},
            )

            # Create the service
            return PlannerService(
                model=self._model, config=config, trace_manager=self._config.get("trace_manager")
            )

        return create_service()


__all__ = [
    "PlannerServiceBuilder",
]
