"""
Configuration module for Saplings planner.

This module defines the configuration classes for the planner module.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BudgetStrategy(str, Enum):
    """Strategy for budget allocation."""

    EQUAL = "equal"  # Equal budget for all steps
    PROPORTIONAL = "proportional"  # Budget proportional to step complexity
    DYNAMIC = "dynamic"  # Dynamically adjust budget based on previous steps
    FIXED = "fixed"  # Fixed budget per step type


class OptimizationStrategy(str, Enum):
    """Strategy for plan optimization."""

    COST = "cost"  # Optimize for minimum cost
    QUALITY = "quality"  # Optimize for maximum quality
    BALANCED = "balanced"  # Balance cost and quality
    SPEED = "speed"  # Optimize for minimum execution time


class CostHeuristicConfig(BaseModel):
    """Configuration for cost heuristics."""

    token_cost_multiplier: float = Field(
        1.0, description="Multiplier for token-based cost estimates"
    )
    base_cost_per_step: float = Field(0.01, description="Base cost for each plan step in USD")
    complexity_factor: float = Field(
        1.5, description="Factor to multiply cost by for each complexity level"
    )
    tool_use_cost: float = Field(
        0.05, description="Additional cost for steps that use tools in USD"
    )
    retrieval_cost_per_doc: float = Field(0.001, description="Cost per document retrieved in USD")
    max_cost_per_step: float = Field(
        1.0, description="Maximum cost allowed for a single step in USD"
    )


class PlannerConfig(BaseModel):
    """Configuration for the planner module."""

    budget_strategy: BudgetStrategy = Field(
        BudgetStrategy.PROPORTIONAL, description="Strategy for budget allocation"
    )
    optimization_strategy: OptimizationStrategy = Field(
        OptimizationStrategy.BALANCED, description="Strategy for plan optimization"
    )
    max_steps: int = Field(10, description="Maximum number of steps in a plan")
    min_steps: int = Field(1, description="Minimum number of steps in a plan")
    total_budget: float = Field(1.0, description="Total budget for the plan in USD")
    allow_budget_overflow: bool = Field(False, description="Whether to allow exceeding the budget")
    budget_overflow_margin: float = Field(
        0.1, description="Margin by which the budget can be exceeded (as a fraction)"
    )
    cost_heuristics: CostHeuristicConfig = Field(
        default_factory=CostHeuristicConfig, description="Cost heuristic configuration"
    )
    enable_pruning: bool = Field(True, description="Whether to enable pruning of unnecessary steps")
    enable_parallelization: bool = Field(
        True, description="Whether to enable parallel execution of independent steps"
    )
    enable_caching: bool = Field(True, description="Whether to enable caching of step results")
    cache_dir: Optional[str] = Field(None, description="Directory to cache plan results")

    @classmethod
    def default(cls) -> "PlannerConfig":
        """
        Create a default configuration.

        Returns:
            PlannerConfig: Default configuration
        """
        return cls()

    @classmethod
    def minimal(cls) -> "PlannerConfig":
        """
        Create a minimal configuration with only essential features enabled.

        Returns:
            PlannerConfig: Minimal configuration
        """
        return cls(
            budget_strategy=BudgetStrategy.EQUAL,
            optimization_strategy=OptimizationStrategy.COST,
            max_steps=5,
            total_budget=0.5,
            enable_pruning=False,
            enable_parallelization=False,
            enable_caching=False,
        )

    @classmethod
    def comprehensive(cls) -> "PlannerConfig":
        """
        Create a comprehensive configuration with all features enabled.

        Returns:
            PlannerConfig: Comprehensive configuration
        """
        return cls(
            budget_strategy=BudgetStrategy.DYNAMIC,
            optimization_strategy=OptimizationStrategy.BALANCED,
            max_steps=20,
            total_budget=2.0,
            allow_budget_overflow=True,
            budget_overflow_margin=0.2,
            cost_heuristics=CostHeuristicConfig(
                token_cost_multiplier=1.2,
                base_cost_per_step=0.02,
                complexity_factor=2.0,
                tool_use_cost=0.1,
                retrieval_cost_per_doc=0.002,
                max_cost_per_step=2.0,
            ),
            enable_pruning=True,
            enable_parallelization=True,
            enable_caching=True,
            cache_dir="./cache/planner",
        )

    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> "PlannerConfig":
        """
        Create a configuration from command-line arguments.

        Args:
            args: Command-line arguments

        Returns:
            PlannerConfig: Configuration
        """
        config = cls()

        if "planner_budget_strategy" in args:
            config.budget_strategy = BudgetStrategy(args["planner_budget_strategy"])

        if "planner_optimization" in args:
            config.optimization_strategy = OptimizationStrategy(args["planner_optimization"])

        if "planner_max_steps" in args:
            config.max_steps = args["planner_max_steps"]

        if "planner_budget" in args:
            config.total_budget = args["planner_budget"]

        if "planner_allow_overflow" in args:
            config.allow_budget_overflow = args["planner_allow_overflow"]

        if "planner_enable_pruning" in args:
            config.enable_pruning = args["planner_enable_pruning"]

        if "planner_enable_parallel" in args:
            config.enable_parallelization = args["planner_enable_parallel"]

        if "planner_enable_cache" in args:
            config.enable_caching = args["planner_enable_cache"]

        if "planner_cache_dir" in args:
            config.cache_dir = args["planner_cache_dir"]

        return config
