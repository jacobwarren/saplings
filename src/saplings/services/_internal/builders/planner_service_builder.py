from __future__ import annotations

"""
Planner Service Builder module for Saplings.

This module provides a builder class for creating PlannerService instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import Any

from saplings.core._internal.builder import ServiceBuilder
from saplings.planner._internal.config import BudgetStrategy, OptimizationStrategy, PlannerConfig
from saplings.services._internal.providers.planner_service import PlannerService

logger = logging.getLogger(__name__)


class PlannerServiceBuilder(ServiceBuilder[PlannerService]):
    """
    Builder for PlannerService.

    This class provides a fluent interface for building PlannerService instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for PlannerService
    builder = PlannerServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_budget_strategy("token_count") \
                    .with_total_budget(1.0) \
                    .with_trace_manager(trace_manager) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the planner service builder."""
        super().__init__(PlannerService)
        self._model = None
        self._budget_strategy = BudgetStrategy.PROPORTIONAL
        self._total_budget = 1.0
        self._allow_budget_overflow = False
        self._budget_overflow_margin = 0.1
        self._optimization_strategy = OptimizationStrategy.BALANCED
        self._max_steps = 10
        self._min_steps = 1
        self._enable_pruning = True
        self._enable_parallelization = True
        self._enable_caching = True
        self._cache_dir = None
        self._trace_manager = None
        # Model is optional for lazy initialization
        # self.require_dependency("model")

    def with_model(self, model: Any) -> PlannerServiceBuilder:
        """
        Set the model.

        Args:
        ----
            model: Model for planning

        Returns:
        -------
            The builder instance for method chaining

        """
        self._model = model
        self.with_dependency("model", model)
        return self

    def with_budget_strategy(self, strategy: str) -> PlannerServiceBuilder:
        """
        Set the budget strategy.

        Args:
        ----
            strategy: Budget strategy (e.g., "proportional", "fixed")

        Returns:
        -------
            The builder instance for method chaining

        """
        self._budget_strategy = BudgetStrategy(strategy)
        return self

    def with_total_budget(self, budget: float) -> PlannerServiceBuilder:
        """
        Set the total budget.

        Args:
        ----
            budget: Total budget for planning

        Returns:
        -------
            The builder instance for method chaining

        """
        self._total_budget = budget
        return self

    def with_allow_budget_overflow(self, allow: bool) -> PlannerServiceBuilder:
        """
        Set whether to allow budget overflow.

        Args:
        ----
            allow: Whether to allow budget overflow

        Returns:
        -------
            The builder instance for method chaining

        """
        self._allow_budget_overflow = allow
        return self

    def with_max_steps(self, max_steps: int) -> PlannerServiceBuilder:
        """
        Set the maximum number of steps.

        Args:
        ----
            max_steps: Maximum number of steps

        Returns:
        -------
            The builder instance for method chaining

        """
        self._max_steps = max_steps
        return self

    def with_trace_manager(self, trace_manager: Any) -> PlannerServiceBuilder:
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

    def build(self) -> PlannerService:
        """
        Build the planner service instance with the configured dependencies.

        Returns
        -------
            The initialized planner service instance

        """
        # Create the planner config
        config = PlannerConfig(
            budget_strategy=self._budget_strategy,
            total_budget=self._total_budget,
            allow_budget_overflow=self._allow_budget_overflow,
            budget_overflow_margin=self._budget_overflow_margin,
            optimization_strategy=self._optimization_strategy,
            max_steps=self._max_steps,
            min_steps=self._min_steps,
            enable_pruning=self._enable_pruning,
            enable_parallelization=self._enable_parallelization,
            enable_caching=self._enable_caching,
            cache_dir=self._cache_dir,
        )
        self.with_dependency("config", config)

        return super().build()
