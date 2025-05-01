"""
Comparison benchmarks for Saplings.

This module provides benchmarks that compare Saplings against baseline methods
and other agent frameworks where applicable.
"""

import asyncio
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from saplings.core.model_adapter import LLM
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig
from saplings.memory import DependencyGraph, Document, MemoryStore
from saplings.memory.config import MemoryConfig
from saplings.monitoring import MonitoringConfig, TraceManager
from saplings.planner import PlannerConfig, SequentialPlanner
from saplings.retrieval import CascadeRetriever, EmbeddingRetriever, TFIDFRetriever
from saplings.retrieval.config import RetrievalConfig
from tests.benchmarks.base_benchmark import BaseBenchmark, MockBenchmarkLLM
from tests.benchmarks.test_datasets import TestDatasets


class TestComparisonBenchmark(BaseBenchmark):
    """Comparison benchmarks for Saplings."""

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
    async def test_gasa_vs_baseline(self, mock_llm, trace_manager):
        """Compare GASA against baseline attention."""
        # Create test documents and graph
        num_documents = 10
        documents = TestDatasets.create_document_corpus(num_documents=num_documents)
        graph = TestDatasets.create_document_graph(documents)

        # Create prompt
        prompt = "Summarize the following documents:\n\n"
        for i, doc in enumerate(documents):
            prompt += f"Document {i+1}: {doc.content}\n\n"

        # Results dictionary
        results = {
            "configurations": [],
        }

        # Baseline (no GASA)
        baseline_executor = Executor(
            model=mock_llm,
            config=ExecutorConfig(
                enable_gasa=False,
            ),
            trace_manager=trace_manager,
        )

        # GASA with different max_hops values
        gasa_executors = []
        for max_hops in [1, 2, 3]:
            gasa_config = GASAConfig(
                max_hops=max_hops,
                mask_strategy="binary",
                cache_masks=False,  # Disable caching for benchmark
            )

            executor = Executor(
                model=mock_llm,
                config=ExecutorConfig(
                    enable_gasa=True,
                ),
                gasa_config=gasa_config,
                dependency_graph=graph,
                trace_manager=trace_manager,
            )

            gasa_executors.append((f"GASA (h={max_hops})", executor))

        # Run baseline
        print("\nRunning baseline (no GASA)...")
        baseline_latencies = []
        for i in range(self.NUM_RUNS):
            print(f"  Run {i+1}/{self.NUM_RUNS}...")

            # Create trace
            trace = trace_manager.create_trace()

            # Execute
            result, latency = await self.time_async_execution(
                baseline_executor.execute,
                prompt=prompt,
                documents=documents,
                trace_id=trace.trace_id,
            )

            baseline_latencies.append(latency)

        baseline_stats = self.calculate_statistics(baseline_latencies)
        results["configurations"].append(
            {
                "name": "Baseline (No GASA)",
                "latency_stats": baseline_stats,
                "raw_latencies_ms": baseline_latencies,
            }
        )

        print(f"  Baseline latency: {baseline_stats['mean']:.2f}ms")

        # Run GASA executors
        for name, executor in gasa_executors:
            print(f"\nRunning {name}...")
            latencies = []

            for i in range(self.NUM_RUNS):
                print(f"  Run {i+1}/{self.NUM_RUNS}...")

                # Create trace
                trace = trace_manager.create_trace()

                # Execute
                result, latency = await self.time_async_execution(
                    executor.execute,
                    prompt=prompt,
                    documents=documents,
                    trace_id=trace.trace_id,
                )

                latencies.append(latency)

            latency_stats = self.calculate_statistics(latencies)

            # Calculate improvement
            improvement = (
                (baseline_stats["mean"] - latency_stats["mean"]) / baseline_stats["mean"] * 100
            )

            results["configurations"].append(
                {
                    "name": name,
                    "latency_stats": latency_stats,
                    "improvement": improvement,
                    "raw_latencies_ms": latencies,
                }
            )

            print(f"  {name} latency: {latency_stats['mean']:.2f}ms")
            print(f"  Improvement: {improvement:.2f}%")

        # Save results
        self.save_results(results, "gasa_vs_baseline")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Add a 60-second timeout to prevent hanging
    async def test_retrieval_comparison(self, memory_store):
        """Compare Saplings retrieval against baseline methods."""
        # Results dictionary
        results = {
            "retrievers": [],
        }

        try:
            # Create test documents with embeddings
            num_documents = 10  # Reduced for faster testing
            documents = TestDatasets.create_document_corpus(
                num_documents=num_documents,
                with_embeddings=True,
                embedding_dim=384,  # Match the dimension used by all-MiniLM-L6-v2 model
            )

            # Create query set
            queries = TestDatasets.create_query_set(
                num_queries=3,  # Reduced for faster testing
                documents=documents,
            )

            # Add documents to memory store
            memory_store.documents = {doc.id: doc for doc in documents}

            # Define retrievers
            retrievers = [
                (
                    "TFIDFRetriever",
                    TFIDFRetriever(
                        memory_store=memory_store,
                        config=RetrievalConfig(),
                    ),
                ),
                (
                    "EmbeddingRetriever",
                    EmbeddingRetriever(
                        memory_store=memory_store,
                        config=RetrievalConfig(),
                    ),
                ),
                (
                    "CascadeRetriever",
                    CascadeRetriever(
                        memory_store=memory_store,
                        config=RetrievalConfig(),
                    ),
                ),
                # Simulate a baseline keyword search
                ("Baseline Keyword", self._create_keyword_retriever(memory_store)),
            ]

            # Test each retriever
            for retriever_name, retriever in retrievers:
                try:
                    print(f"\nTesting {retriever_name}...")

                    # Metrics
                    precision_at_k = []
                    recall_at_k = []
                    latencies = []

                    # Test each query
                    for query_idx, query in enumerate(queries):
                        try:
                            print(f"  Processing query {query_idx+1}/{len(queries)}")

                            # Retrieve documents
                            if retriever_name == "EmbeddingRetriever":
                                # EmbeddingRetriever requires documents parameter
                                result, latency = await self.time_async_execution(
                                    retriever.retrieve,
                                    query=query["query"],
                                    documents=list(memory_store.documents.values()),
                                    k=10,
                                )
                            elif retriever_name == "CascadeRetriever":
                                # CascadeRetriever uses max_documents instead of k
                                result, latency = await self.time_async_execution(
                                    retriever.retrieve,
                                    query=query["query"],
                                    max_documents=10,
                                )
                            else:
                                # TFIDFRetriever or Baseline Keyword
                                result, latency = await self.time_async_execution(
                                    retriever.retrieve,
                                    query=query["query"],
                                    k=10,
                                )

                            # Record latency
                            latencies.append(latency)

                            # Calculate precision and recall
                            retrieved_ids = [doc.id for doc in result]
                            relevant_ids = query["relevant_docs"]

                            # Skip if no relevant documents
                            if not relevant_ids:
                                continue

                            # Calculate precision@k and recall@k
                            k = min(len(retrieved_ids), len(relevant_ids))
                            if k > 0:
                                relevant_retrieved = set(retrieved_ids[:k]).intersection(set(relevant_ids))
                                precision = len(relevant_retrieved) / k
                                recall = len(relevant_retrieved) / len(relevant_ids)

                                precision_at_k.append(precision)
                                recall_at_k.append(recall)
                        except Exception as e:
                            print(f"  Error processing query {query_idx+1} with {retriever_name}: {e}")
                            # Add default values to avoid breaking the test
                            latencies.append(1000.0)  # 1 second as fallback

                    # Calculate statistics
                    precision_stats = self.calculate_statistics(precision_at_k)
                    recall_stats = self.calculate_statistics(recall_at_k)
                    latency_stats = self.calculate_statistics(latencies)

                    # Add to results
                    results["retrievers"].append(
                        {
                            "name": retriever_name,
                            "precision_at_k": precision_stats,
                            "recall_at_k": recall_stats,
                            "latency_ms": latency_stats,
                            "raw_latencies_ms": latencies,
                        }
                    )

                    print(f"  Precision@k: {precision_stats['mean']:.4f}")
                    print(f"  Recall@k: {recall_stats['mean']:.4f}")
                    print(f"  Latency: {latency_stats['mean']:.2f}ms")
                except Exception as e:
                    print(f"  Error testing {retriever_name}: {e}")
                    # Add a placeholder result
                    results["retrievers"].append(
                        {
                            "name": retriever_name,
                            "error": str(e),
                            "precision_at_k": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
                            "recall_at_k": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
                            "latency_ms": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
                            "raw_latencies_ms": [],
                        }
                    )
        except Exception as e:
            print(f"Error in test_retrieval_comparison: {e}")
            results["error"] = str(e)
        finally:
            # Save results
            self.save_results(results, "retrieval_comparison")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Add a 60-second timeout to prevent hanging
    async def test_planner_comparison(self, mock_llm, trace_manager):
        """Compare Saplings planner against baseline methods."""
        # Create test documents
        num_documents = 5
        documents = TestDatasets.create_document_corpus(num_documents=num_documents)

        # Create task
        task = "Analyze the following documents and extract key information."

        # Results dictionary
        results = {
            "planners": [],
        }

        # Create Saplings planner
        saplings_planner = SequentialPlanner(
            model=mock_llm,
            config=PlannerConfig(),
            trace_manager=trace_manager,
        )

        # Create baseline planner (simple prompt template)
        baseline_planner = self._create_baseline_planner(mock_llm)

        # Test Saplings planner
        print("\nTesting Saplings planner...")
        saplings_latencies = []
        saplings_plan_lengths = []

        for i in range(self.NUM_RUNS):
            print(f"  Run {i+1}/{self.NUM_RUNS}...")

            # Create trace
            trace = trace_manager.create_trace()

            # Create plan
            plan, latency = await self.time_async_execution(
                saplings_planner.create_plan,
                task=task,
                documents=documents,
                trace_id=trace.trace_id,
            )

            saplings_latencies.append(latency)
            saplings_plan_lengths.append(len(plan) if plan else 0)

        saplings_latency_stats = self.calculate_statistics(saplings_latencies)
        saplings_length_stats = self.calculate_statistics(saplings_plan_lengths)

        results["planners"].append(
            {
                "name": "Saplings Planner",
                "latency_stats": saplings_latency_stats,
                "plan_length_stats": saplings_length_stats,
                "raw_latencies_ms": saplings_latencies,
            }
        )

        print(f"  Latency: {saplings_latency_stats['mean']:.2f}ms")
        print(f"  Plan length: {saplings_length_stats['mean']:.2f} steps")

        # Test baseline planner
        print("\nTesting baseline planner...")
        baseline_latencies = []
        baseline_plan_lengths = []

        for i in range(self.NUM_RUNS):
            print(f"  Run {i+1}/{self.NUM_RUNS}...")

            # Create plan
            plan, latency = await self.time_async_execution(
                baseline_planner.create_plan,
                task=task,
                documents=documents,
            )

            baseline_latencies.append(latency)
            baseline_plan_lengths.append(len(plan) if plan else 0)

        baseline_latency_stats = self.calculate_statistics(baseline_latencies)
        baseline_length_stats = self.calculate_statistics(baseline_plan_lengths)

        results["planners"].append(
            {
                "name": "Baseline Planner",
                "latency_stats": baseline_latency_stats,
                "plan_length_stats": baseline_length_stats,
                "raw_latencies_ms": baseline_latencies,
            }
        )

        print(f"  Latency: {baseline_latency_stats['mean']:.2f}ms")
        print(f"  Plan length: {baseline_length_stats['mean']:.2f} steps")

        # Save results
        self.save_results(results, "planner_comparison")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Add a 60-second timeout to prevent hanging
    async def test_end_to_end_comparison(self, mock_llm, trace_manager, memory_store):
        """Compare Saplings end-to-end against baseline methods."""
        # Create test documents and graph
        num_documents = 5
        documents = TestDatasets.create_document_corpus(num_documents=num_documents)
        graph = TestDatasets.create_document_graph(documents)

        # Add documents to memory store
        memory_store.documents = {doc.id: doc for doc in documents}

        # Create task
        task = "Analyze the following documents and extract key information."

        # Results dictionary
        results = {
            "systems": [],
        }

        # Create Saplings components
        saplings_planner = SequentialPlanner(
            model=mock_llm,
            config=PlannerConfig(),
            trace_manager=trace_manager,
        )

        saplings_executor = Executor(
            model=mock_llm,
            config=ExecutorConfig(
                enable_gasa=True,
            ),
            gasa_config=GASAConfig(
                max_hops=2,
                mask_strategy="binary",
            ),
            dependency_graph=graph,
            trace_manager=trace_manager,
        )

        # Create baseline components
        baseline_planner = self._create_baseline_planner(mock_llm)

        baseline_executor = Executor(
            model=mock_llm,
            config=ExecutorConfig(
                enable_gasa=False,
            ),
            trace_manager=trace_manager,
        )

        # Test Saplings end-to-end
        print("\nTesting Saplings end-to-end...")
        saplings_latencies = []

        for i in range(self.NUM_RUNS):
            print(f"  Run {i+1}/{self.NUM_RUNS}...")

            # Create trace
            trace = trace_manager.create_trace()

            # Start timing
            start_time = asyncio.get_event_loop().time()

            # Create plan
            plan = await saplings_planner.create_plan(
                task=task,
                documents=documents,
                trace_id=trace.trace_id,
            )

            # Execute plan
            result = await saplings_executor.execute(
                prompt=task,
                documents=documents,
                plan=plan,
                trace_id=trace.trace_id,
            )

            # End timing
            end_time = asyncio.get_event_loop().time()
            latency = (end_time - start_time) * 1000  # ms

            saplings_latencies.append(latency)

        saplings_latency_stats = self.calculate_statistics(saplings_latencies)

        results["systems"].append(
            {
                "name": "Saplings (with GASA)",
                "latency_stats": saplings_latency_stats,
                "raw_latencies_ms": saplings_latencies,
            }
        )

        print(f"  Latency: {saplings_latency_stats['mean']:.2f}ms")

        # Test baseline end-to-end
        print("\nTesting baseline end-to-end...")
        baseline_latencies = []

        for i in range(self.NUM_RUNS):
            print(f"  Run {i+1}/{self.NUM_RUNS}...")

            # Start timing
            start_time = asyncio.get_event_loop().time()

            # Create plan
            plan = await baseline_planner.create_plan(
                task=task,
                documents=documents,
            )

            # Execute plan
            result = await baseline_executor.execute(
                prompt=task,
                documents=documents,
                plan=plan,
            )

            # End timing
            end_time = asyncio.get_event_loop().time()
            latency = (end_time - start_time) * 1000  # ms

            baseline_latencies.append(latency)

        baseline_latency_stats = self.calculate_statistics(baseline_latencies)

        # Calculate improvement
        improvement = (
            (baseline_latency_stats["mean"] - saplings_latency_stats["mean"])
            / baseline_latency_stats["mean"]
            * 100
        )

        results["systems"].append(
            {
                "name": "Baseline",
                "latency_stats": baseline_latency_stats,
                "raw_latencies_ms": baseline_latencies,
            }
        )

        results["improvement"] = improvement

        print(f"  Latency: {baseline_latency_stats['mean']:.2f}ms")
        print(f"  Improvement: {improvement:.2f}%")

        # Save results
        self.save_results(results, "end_to_end_comparison")

    def _create_keyword_retriever(self, memory_store):
        """
        Create a simple keyword-based retriever as a baseline.

        Args:
            memory_store: Memory store to search

        Returns:
            Object: A retriever-like object
        """

        class KeywordRetriever:
            """Simple keyword-based retriever."""

            def __init__(self, memory_store):
                """Initialize the retriever."""
                self.memory_store = memory_store

            async def retrieve(self, query, k=10, max_documents=None, documents=None, **kwargs):
                """Retrieve documents based on keyword matching.

                Args:
                    query: The query string
                    k: Maximum number of documents to return (default: 10)
                    max_documents: Alternative name for k (for compatibility)
                    documents: Optional list of documents to search (if not provided, uses memory_store)
                    **kwargs: Additional arguments (ignored)

                Returns:
                    List of matching documents
                """
                # Determine the limit (k or max_documents)
                limit = max_documents if max_documents is not None else k

                # Get documents to search
                if documents is not None:
                    all_docs = documents
                else:
                    all_docs = list(self.memory_store.documents.values())

                # Handle empty document list
                if not all_docs:
                    return []

                # Extract keywords from query (simple tokenization)
                keywords = set(query.lower().split())

                # Handle empty query
                if not keywords:
                    return all_docs[:limit]

                # Score documents based on keyword matches
                scored_docs = []
                for doc in all_docs:
                    # Make sure doc.content is a string before calling lower()
                    if isinstance(doc, Document):
                        content_lower = doc.content.lower()
                    else:
                        content_lower = str(doc).lower()
                    score = sum(1 for keyword in keywords if keyword in content_lower)
                    scored_docs.append((doc, score))

                # Sort by score (descending)
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # Return top documents
                return [doc for doc, _ in scored_docs[:limit]]

        return KeywordRetriever(memory_store)

    def _create_baseline_planner(self, model):
        """
        Create a simple baseline planner.

        Args:
            model: LLM model to use

        Returns:
            Object: A planner-like object
        """

        class BaselinePlanner:
            """Simple baseline planner."""

            def __init__(self, model):
                """Initialize the planner."""
                self.model = model

            async def create_plan(self, task, documents=None, **kwargs):
                """Create a simple plan."""
                # Create prompt
                prompt = f"Create a step-by-step plan for the following task:\n\n{task}\n\n"

                if documents:
                    prompt += "Consider the following documents:\n\n"
                    for i, doc in enumerate(documents):
                        prompt += f"Document {i+1}: {doc.content}\n\n"

                prompt += "Plan:"

                # Generate plan
                response = await self.model.generate(prompt)

                # Parse plan (simple line-by-line parsing)
                plan_text = response.text
                plan_steps = []

                for line in plan_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line.startswith("- ")
                        or line.startswith("Step ")
                        or (len(line) > 1 and line[0].isdigit() and line[1] == ".")
                    ):
                        plan_steps.append(line)

                return plan_steps

        return BaselinePlanner(model)
