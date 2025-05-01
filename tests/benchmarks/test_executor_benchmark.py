"""
Benchmark tests for the executor component.

This module provides benchmark tests for the executor component, measuring
execution time, memory usage, and quality improvements with different strategies.
"""

import asyncio
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig, RefinementStrategy, VerificationStrategy
from saplings.gasa import GASAConfig
from saplings.memory import Document, MemoryStore
from saplings.memory.config import MemoryConfig
from saplings.monitoring import MonitoringConfig, TraceManager
from saplings.planner import PlannerConfig, PlanStep, SequentialPlanner
from tests.benchmarks.base_benchmark import BaseBenchmark, MockBenchmarkLLM
from tests.benchmarks.test_datasets import TestDatasets


class TestExecutorBenchmark(BaseBenchmark):
    """Benchmark tests for the executor component.

    Note: These tests use the TestDatasets class which contains code samples with
    potential infinite recursion bugs. The code samples have been modified to include
    execution guards to prevent actual infinite recursion during testing.
    """

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockBenchmarkLLM("mock://model/latest", response_time_ms=300)

    @pytest.fixture
    def trace_manager(self):
        """Create a trace manager for testing."""
        return TraceManager(config=MonitoringConfig())

    @pytest.fixture
    def memory_store(self):
        """Create a memory store for testing."""
        return MemoryStore(config=MemoryConfig())

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Add a 60-second timeout to prevent hanging
    async def test_executor_strategies(self, mock_llm, trace_manager):
        """Test executor performance with different strategies."""
        # Create test documents
        num_documents = 5
        documents = TestDatasets.create_document_corpus(num_documents=num_documents)

        # Create prompt
        prompt = "Summarize the following documents:\n\n"
        for i, doc in enumerate(documents):
            prompt += f"Document {i+1}: {doc.content}\n\n"

        # Results dictionary
        results = {
            "configurations": [],
        }

        # Test with different configurations
        configurations = [
            {
                "name": "Baseline",
                "config": ExecutorConfig(),
            },
            {
                "name": "With Refinement",
                "config": ExecutorConfig(
                    refinement_strategy=RefinementStrategy.ITERATIVE,
                    max_refinement_steps=2,
                ),
            },
            {
                "name": "With Verification",
                "config": ExecutorConfig(
                    verification_strategy=VerificationStrategy.BASIC,
                    verification_threshold=0.7,
                ),
            },
            {
                "name": "With Speculative Execution",
                "config": ExecutorConfig(
                    enable_speculative_execution=True,
                    speculation_threshold=0.7,
                ),
            },
        ]

        for config_info in configurations:
            print(f"\nTesting configuration: {config_info['name']}...")

            # Create executor
            executor = Executor(
                model=mock_llm,
                config=config_info["config"],
                trace_manager=trace_manager,
            )

            # Run multiple times and measure
            latencies = []
            token_counts = []
            memory_usages = []

            for i in range(self.NUM_RUNS):
                print(f"  Run {i+1}/{self.NUM_RUNS}...")

                # Create trace
                trace = trace_manager.create_trace()

                # Measure memory before
                memory_before = self.get_memory_usage()

                # Execute
                result, latency = await self.time_async_execution(
                    executor.execute,
                    prompt=prompt,
                    documents=documents,
                    trace_id=trace.trace_id,
                )

                # Measure memory after
                memory_after = self.get_memory_usage()

                # Calculate metrics
                token_count = result.token_count if hasattr(result, "token_count") else 0
                memory_usage = memory_after - memory_before

                # Record metrics
                latencies.append(latency)
                token_counts.append(token_count)
                memory_usages.append(memory_usage)

            # Calculate statistics
            latency_stats = self.calculate_statistics(latencies)
            token_count_stats = self.calculate_statistics(token_counts)
            memory_usage_stats = self.calculate_statistics(memory_usages)

            # Add to results
            results["configurations"].append(
                {
                    "name": config_info["name"],
                    "latency_stats": latency_stats,
                    "token_count_stats": token_count_stats,
                    "memory_usage_stats": memory_usage_stats,
                    "raw_latencies_ms": latencies,
                }
            )

            print(f"  Latency: {latency_stats['mean']:.2f}ms")
            print(f"  Token count: {token_count_stats['mean']:.2f}")
            print(f"  Memory usage: {memory_usage_stats['mean']:.2f}MB")

        # Save results
        self.save_results(results, "executor_strategies")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Add a 60-second timeout to prevent hanging
    async def test_executor_planner_integration(self, mock_llm, trace_manager):
        """Test executor performance with planner integration."""
        # Create test documents
        num_documents = 5
        documents = TestDatasets.create_document_corpus(num_documents=num_documents)

        # Create task
        task = "Analyze the following documents and extract key information."

        # Results dictionary
        results = {
            "configurations": [],
        }

        # Test with different configurations
        configurations = [
            {
                "name": "Direct Execution",
                "use_planner": False,
            },
            {
                "name": "With Planner",
                "use_planner": True,
            },
        ]

        for config_info in configurations:
            print(f"\nTesting configuration: {config_info['name']}...")

            # Create executor
            executor = Executor(
                model=mock_llm,
                config=ExecutorConfig(),
                trace_manager=trace_manager,
            )

            # Create planner if needed
            planner = None
            if config_info["use_planner"]:
                planner = SequentialPlanner(
                    model=mock_llm,
                    config=PlannerConfig(),
                    trace_manager=trace_manager,
                )

            # Run multiple times and measure
            latencies = []
            token_counts = []
            memory_usages = []

            for i in range(self.NUM_RUNS):
                print(f"  Run {i+1}/{self.NUM_RUNS}...")

                # Create trace
                trace = trace_manager.create_trace()

                # Measure memory before
                memory_before = self.get_memory_usage()

                # Start timing
                start_time = asyncio.get_event_loop().time()

                # Create plan if using planner
                plan = None
                if planner:
                    plan = await planner.create_plan(
                        task=task,
                        documents=documents,
                        trace_id=trace.trace_id,
                    )

                # Execute
                if plan:
                    # Execute with plan
                    result = await executor.execute(
                        prompt=task,
                        documents=documents,
                        plan=plan,
                        trace_id=trace.trace_id,
                    )
                else:
                    # Direct execution
                    result = await executor.execute(
                        prompt=task,
                        documents=documents,
                        trace_id=trace.trace_id,
                    )

                # End timing
                end_time = asyncio.get_event_loop().time()
                latency = (end_time - start_time) * 1000  # ms

                # Measure memory after
                memory_after = self.get_memory_usage()

                # Calculate metrics
                token_count = result.token_count if hasattr(result, "token_count") else 0
                memory_usage = memory_after - memory_before

                # Record metrics
                latencies.append(latency)
                token_counts.append(token_count)
                memory_usages.append(memory_usage)

            # Calculate statistics
            latency_stats = self.calculate_statistics(latencies)
            token_count_stats = self.calculate_statistics(token_counts)
            memory_usage_stats = self.calculate_statistics(memory_usages)

            # Add to results
            results["configurations"].append(
                {
                    "name": config_info["name"],
                    "use_planner": config_info["use_planner"],
                    "latency_stats": latency_stats,
                    "token_count_stats": token_count_stats,
                    "memory_usage_stats": memory_usage_stats,
                    "raw_latencies_ms": latencies,
                }
            )

            print(f"  Latency: {latency_stats['mean']:.2f}ms")
            print(f"  Token count: {token_count_stats['mean']:.2f}")
            print(f"  Memory usage: {memory_usage_stats['mean']:.2f}MB")

        # Save results
        self.save_results(results, "executor_planner_integration")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Add a 60-second timeout to prevent hanging
    async def test_executor_streaming(self, mock_llm, trace_manager):
        """Test executor streaming performance."""
        # Create test documents
        num_documents = 3
        documents = TestDatasets.create_document_corpus(num_documents=num_documents)

        # Create prompt
        prompt = "Summarize the following documents:\n\n"
        for i, doc in enumerate(documents):
            prompt += f"Document {i+1}: {doc.content}\n\n"

        # Results dictionary
        results = {
            "configurations": [],
        }

        # Test with different configurations
        configurations = [
            {
                "name": "Non-Streaming",
                "streaming": False,
            },
            {
                "name": "Streaming",
                "streaming": True,
            },
        ]

        for config_info in configurations:
            print(f"\nTesting configuration: {config_info['name']}...")

            # Create executor
            executor = Executor(
                model=mock_llm,
                config=ExecutorConfig(),
                trace_manager=trace_manager,
            )

            # Run multiple times and measure
            latencies = []
            token_counts = []
            first_token_latencies = []

            for i in range(self.NUM_RUNS):
                print(f"  Run {i+1}/{self.NUM_RUNS}...")

                # Create trace
                trace = trace_manager.create_trace()

                # Start timing
                start_time = asyncio.get_event_loop().time()

                if config_info["streaming"]:
                    # Streaming execution
                    first_token_time = None
                    token_count = 0

                    async for chunk in executor.model.generate_streaming(
                        prompt=prompt,
                    ):
                        # Record time of first token
                        if first_token_time is None:
                            first_token_time = asyncio.get_event_loop().time()

                        # Count tokens (approximate)
                        token_count += len(chunk.split())

                    # Calculate first token latency
                    if first_token_time:
                        first_token_latency = (first_token_time - start_time) * 1000  # ms
                        first_token_latencies.append(first_token_latency)
                else:
                    # Non-streaming execution
                    result = await executor.execute(
                        prompt=prompt,
                        documents=documents,
                        trace_id=trace.trace_id,
                    )

                    # Approximate token count
                    token_count = (
                        result.token_count
                        if hasattr(result, "token_count")
                        else len(result.text.split())
                    )

                    # No first token latency for non-streaming
                    first_token_latencies.append(0)

                # End timing
                end_time = asyncio.get_event_loop().time()
                latency = (end_time - start_time) * 1000  # ms

                # Record metrics
                latencies.append(latency)
                token_counts.append(token_count)

            # Calculate statistics
            latency_stats = self.calculate_statistics(latencies)
            token_count_stats = self.calculate_statistics(token_counts)
            first_token_stats = self.calculate_statistics(first_token_latencies)

            # Add to results
            results["configurations"].append(
                {
                    "name": config_info["name"],
                    "streaming": config_info["streaming"],
                    "latency_stats": latency_stats,
                    "token_count_stats": token_count_stats,
                    "first_token_latency_stats": first_token_stats,
                    "raw_latencies_ms": latencies,
                }
            )

            print(f"  Total latency: {latency_stats['mean']:.2f}ms")
            if config_info["streaming"]:
                print(f"  First token latency: {first_token_stats['mean']:.2f}ms")
            print(f"  Token count: {token_count_stats['mean']:.2f}")

        # Save results
        self.save_results(results, "executor_streaming")
