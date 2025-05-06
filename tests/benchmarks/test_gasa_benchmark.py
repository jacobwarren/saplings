from __future__ import annotations

"""
Benchmark tests for GASA.

These tests measure the performance of GASA compared to standard attention.
They are marked as benchmark tests and are skipped by default.
"""


import asyncio
import time

import numpy as np
import pytest

from saplings.core.model_adapter import LLM
from saplings.gasa import GASAConfig, GASAService, MaskFormat, MaskType
from saplings.memory import DependencyGraph, Document, DocumentMetadata

# Try to import vLLM
try:
    import vllm

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# Skip all tests if vLLM is not installed
pytestmark = [
    pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed"),
    pytest.mark.benchmark,  # Mark as benchmark test
]


def create_test_documents(count: int) -> list[Document]:
    """Create test documents."""
    docs = []
    for i in range(count):
        doc = Document(
            id=f"doc{i}",
            content=f"Document {i} about artificial intelligence and machine learning. " * 10,
            metadata=DocumentMetadata(source=f"doc{i}.txt"),
        )
        docs.append(doc)
    return docs


def create_test_graph(docs: list[Document]) -> DependencyGraph:
    """Create a test dependency graph."""
    graph = DependencyGraph()

    # Add documents to graph
    for doc in docs:
        graph.add_document_node(doc)

    # Add relationships (connect documents in a chain)
    for i in range(len(docs) - 1):
        graph.add_relationship(docs[i].id, docs[i + 1].id, "relates_to", 0.9)

    return graph


class TestGASABenchmark:
    """Benchmark tests for GASA."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create test documents
        self.docs = create_test_documents(5)

        # Create dependency graph
        self.graph = create_test_graph(self.docs)

        # Create prompt
        self.prompt = "Summarize the following documents:\n\n"
        for i, doc in enumerate(self.docs):
            self.prompt += f"Document {i}: {doc.content}\n\n"
        self.prompt += "Summary:"

    @pytest.mark.parametrize("max_hops", [0, 1, 2, 3])
    def test_mask_building_benchmark(self, max_hops) -> None:
        """Benchmark mask building with different max_hops values."""
        # Create GASA configuration
        config = GASAConfig(
            max_hops=max_hops,
            mask_strategy="binary",
            fallback_strategy="block_diagonal",
            visualize=False,
        )

        # Create GASA service
        gasa_service = GASAService(
            graph=self.graph,
            config=config,
        )

        # Measure mask building time
        start_time = time.time()
        gasa_service.build_mask(
            documents=self.docs,
            prompt=self.prompt,
            format=MaskFormat.DENSE,
            mask_type=MaskType.ATTENTION,
        )
        time.time() - start_time

        # Print results

        # No assertions, just performance measurement

    @pytest.mark.parametrize("fallback_strategy", ["block_diagonal", "prompt_composer"])
    def test_fallback_strategy_benchmark(self, fallback_strategy) -> None:
        """Benchmark different fallback strategies."""
        # Create GASA configuration
        config = GASAConfig(
            max_hops=2, mask_strategy="binary", fallback_strategy=fallback_strategy, visualize=False
        )

        # Create GASA service
        gasa_service = GASAService(
            graph=self.graph,
            config=config,
        )

        # Create input IDs and attention mask
        input_ids = list(range(1000))
        attention_mask = np.ones(1000, dtype=np.int32)

        # Measure apply_gasa time
        start_time = time.time()
        gasa_service.apply_gasa(
            documents=self.docs,
            prompt=self.prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_supports_sparse_attention=False,
        )
        time.time() - start_time

        # Print results

        # No assertions, just performance measurement

    @pytest.mark.skipif(True, reason="This test requires a real vLLM model and is slow")
    @pytest.mark.parametrize("use_gasa", [False, True])
    def test_generation_benchmark(self, use_gasa) -> None:
        """Benchmark generation with and without GASA."""
        try:
            # Create a model
            model = LLM.create(
                provider="vllm",
                model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                device="cpu",  # Use CPU for CI testing
            )

            if use_gasa:
                # Create GASA configuration
                config = GASAConfig(
                    max_hops=2,
                    mask_strategy="binary",
                    fallback_strategy="block_diagonal",
                    visualize=False,
                )

                # Create GASA service
                gasa_service = GASAService(
                    graph=self.graph,
                    config=config,
                )

                # Apply GASA to the prompt
                result = gasa_service.apply_gasa(
                    documents=self.docs, prompt=self.prompt, model_supports_sparse_attention=True
                )

                # Generate text with GASA
                start_time = time.time()
                asyncio.run(
                    model.generate(
                        prompt=result["prompt"],
                        attention_mask=result.get("attention_mask"),
                        max_tokens=100,
                    )
                )
                time.time() - start_time
            else:
                # Generate text without GASA
                start_time = time.time()
                asyncio.run(model.generate(prompt=self.prompt, max_tokens=100))
                time.time() - start_time

            # Print results

            # No assertions, just performance measurement
        except Exception as e:
            pytest.skip(f"vLLM model loading failed: {e}")
