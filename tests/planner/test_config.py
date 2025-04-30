"""
Tests for the planner configuration module.
"""

import pytest

from saplings.planner.config import BudgetStrategy, CostHeuristicConfig, OptimizationStrategy, PlannerConfig


class TestCostHeuristicConfig:
    """Tests for the CostHeuristicConfig class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        config = CostHeuristicConfig()
        
        assert config.token_cost_multiplier == 1.0
        assert config.base_cost_per_step == 0.01
        assert config.complexity_factor == 1.5
        assert config.tool_use_cost == 0.05
        assert config.retrieval_cost_per_doc == 0.001
        assert config.max_cost_per_step == 1.0
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        config = CostHeuristicConfig(
            token_cost_multiplier=2.0,
            base_cost_per_step=0.02,
            complexity_factor=2.0,
            tool_use_cost=0.1,
            retrieval_cost_per_doc=0.002,
            max_cost_per_step=2.0,
        )
        
        assert config.token_cost_multiplier == 2.0
        assert config.base_cost_per_step == 0.02
        assert config.complexity_factor == 2.0
        assert config.tool_use_cost == 0.1
        assert config.retrieval_cost_per_doc == 0.002
        assert config.max_cost_per_step == 2.0


class TestPlannerConfig:
    """Tests for the PlannerConfig class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        config = PlannerConfig()
        
        assert config.budget_strategy == BudgetStrategy.PROPORTIONAL
        assert config.optimization_strategy == OptimizationStrategy.BALANCED
        assert config.max_steps == 10
        assert config.min_steps == 1
        assert config.total_budget == 1.0
        assert config.allow_budget_overflow is False
        assert config.budget_overflow_margin == 0.1
        assert isinstance(config.cost_heuristics, CostHeuristicConfig)
        assert config.enable_pruning is True
        assert config.enable_parallelization is True
        assert config.enable_caching is True
        assert config.cache_dir is None
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        config = PlannerConfig(
            budget_strategy=BudgetStrategy.EQUAL,
            optimization_strategy=OptimizationStrategy.COST,
            max_steps=5,
            min_steps=2,
            total_budget=0.5,
            allow_budget_overflow=True,
            budget_overflow_margin=0.2,
            cost_heuristics=CostHeuristicConfig(
                token_cost_multiplier=2.0,
                base_cost_per_step=0.02,
            ),
            enable_pruning=False,
            enable_parallelization=False,
            enable_caching=False,
            cache_dir="/tmp/cache",
        )
        
        assert config.budget_strategy == BudgetStrategy.EQUAL
        assert config.optimization_strategy == OptimizationStrategy.COST
        assert config.max_steps == 5
        assert config.min_steps == 2
        assert config.total_budget == 0.5
        assert config.allow_budget_overflow is True
        assert config.budget_overflow_margin == 0.2
        assert config.cost_heuristics.token_cost_multiplier == 2.0
        assert config.cost_heuristics.base_cost_per_step == 0.02
        assert config.enable_pruning is False
        assert config.enable_parallelization is False
        assert config.enable_caching is False
        assert config.cache_dir == "/tmp/cache"
    
    def test_default_factory(self):
        """Test default factory method."""
        config = PlannerConfig.default()
        
        assert config.budget_strategy == BudgetStrategy.PROPORTIONAL
        assert config.optimization_strategy == OptimizationStrategy.BALANCED
        assert config.max_steps == 10
        assert config.min_steps == 1
        assert config.total_budget == 1.0
    
    def test_minimal_factory(self):
        """Test minimal factory method."""
        config = PlannerConfig.minimal()
        
        assert config.budget_strategy == BudgetStrategy.EQUAL
        assert config.optimization_strategy == OptimizationStrategy.COST
        assert config.max_steps == 5
        assert config.total_budget == 0.5
        assert config.enable_pruning is False
        assert config.enable_parallelization is False
        assert config.enable_caching is False
    
    def test_comprehensive_factory(self):
        """Test comprehensive factory method."""
        config = PlannerConfig.comprehensive()
        
        assert config.budget_strategy == BudgetStrategy.DYNAMIC
        assert config.optimization_strategy == OptimizationStrategy.BALANCED
        assert config.max_steps == 20
        assert config.total_budget == 2.0
        assert config.allow_budget_overflow is True
        assert config.budget_overflow_margin == 0.2
        assert config.cost_heuristics.token_cost_multiplier == 1.2
        assert config.cost_heuristics.base_cost_per_step == 0.02
        assert config.enable_pruning is True
        assert config.enable_parallelization is True
        assert config.enable_caching is True
        assert config.cache_dir == "./cache/planner"
    
    def test_from_cli_args_empty(self):
        """Test from_cli_args method with empty arguments."""
        config = PlannerConfig.from_cli_args({})
        
        assert config.budget_strategy == BudgetStrategy.PROPORTIONAL
        assert config.optimization_strategy == OptimizationStrategy.BALANCED
        assert config.max_steps == 10
        assert config.total_budget == 1.0
    
    def test_from_cli_args_custom(self):
        """Test from_cli_args method with custom arguments."""
        args = {
            "planner_budget_strategy": "equal",
            "planner_optimization": "cost",
            "planner_max_steps": 5,
            "planner_budget": 0.5,
            "planner_allow_overflow": True,
            "planner_enable_pruning": False,
            "planner_enable_parallel": False,
            "planner_enable_cache": False,
            "planner_cache_dir": "/tmp/cache",
        }
        
        config = PlannerConfig.from_cli_args(args)
        
        assert config.budget_strategy == BudgetStrategy.EQUAL
        assert config.optimization_strategy == OptimizationStrategy.COST
        assert config.max_steps == 5
        assert config.total_budget == 0.5
        assert config.allow_budget_overflow is True
        assert config.enable_pruning is False
        assert config.enable_parallelization is False
        assert config.enable_caching is False
        assert config.cache_dir == "/tmp/cache"
