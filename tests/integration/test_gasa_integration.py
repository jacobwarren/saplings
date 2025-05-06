from __future__ import annotations

"""
Integration tests for GASA.

These tests require vLLM to be installed and will be skipped if it's not available.
"""


import asyncio

import numpy as np
import pytest

from saplings.core.model_adapter import LLM
from saplings.gasa import (
    FallbackStrategy,
    GASAConfig,
    GASAService,
    MaskFormat,
    MaskStrategy,
    MaskType,
)
from saplings.memory import DependencyGraph, Document, DocumentMetadata, MemoryStore

# Try to import vLLM
try:
    import importlib.util

    HAS_VLLM = importlib.util.find_spec("vllm") is not None
except ImportError:
    HAS_VLLM = False

# Skip all tests if vLLM is not installed
pytestmark = pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")


class TestGASAIntegration:
    """Test GASA integration with vLLM."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create memory and graph
        self.memory = MemoryStore()
        self.graph = DependencyGraph()

        # Add documents
        self.docs = []
        for i in range(3):
            doc = Document(
                id=f"doc{i}",
                content=f"Document {i} about artificial intelligence.",
                metadata=DocumentMetadata(
                    source=f"doc{i}.txt",
                    content_type="text",
                    language="en",
                    author="test",
                ),
                embedding=np.random.default_rng().random(768).astype(np.float32),
            )
            self.docs.append(doc)
            self.memory.add_document(
                content=doc.content,
                metadata=doc.metadata,
                document_id=doc.id,
                embedding=doc.embedding.tolist() if doc.embedding is not None else None,
            )

        # Add relationships to graph
        self.graph.add_document_node(self.docs[0])
        self.graph.add_document_node(self.docs[1])
        self.graph.add_document_node(self.docs[2])
        self.graph.add_relationship("doc0", "doc1", "relates_to", 0.9)
        self.graph.add_relationship("doc1", "doc2", "relates_to", 0.8)

        # Create GASA configuration
        self.config = GASAConfig(
            max_hops=2,
            mask_strategy=MaskStrategy.BINARY,
            fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
            visualize=False,
            enabled=True,
            global_tokens=["[CLS]", "[SEP]"],
            summary_token="[SUM]",
            add_summary_token=False,
            block_size=512,
            overlap=0,
            soft_mask_temperature=1.0,
            cache_masks=False,
            cache_dir=None,
            visualization_dir=None,
            enable_shadow_model=False,
            shadow_model_name="Qwen/Qwen3-0.6B",
            shadow_model_device="cpu",
            shadow_model_cache_dir=None,
            enable_prompt_composer=False,
            focus_tags=True,
            core_tag="core",
            near_tag="near",
            summary_tag="summary",
        )

        # Create GASA service
        self.gasa_service = GASAService(
            graph=self.graph,
            config=self.config,
        )

        # Create prompt
        self.prompt = "Summarize the following documents:\n\n"
        for i, doc in enumerate(self.docs):
            self.prompt += f"Document {i}: {doc.content}\n\n"
        self.prompt += "Summary:"

    def test_build_mask(self) -> None:
        """Test building a mask with real documents."""
        # Build a mask
        mask = self.gasa_service.build_mask(
            documents=self.docs,
            prompt=self.prompt,
            format=MaskFormat.DENSE,
            mask_type=MaskType.ATTENTION,
        )

        # Verify mask properties
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.shape[0] == mask.shape[1]  # Square matrix
        assert np.all(mask.diagonal() == 1)  # Diagonal should be all 1s

    def test_apply_gasa_with_sparse_attention(self) -> None:
        """Test applying GASA with sparse attention support."""
        # Apply GASA
        result = self.gasa_service.apply_gasa(
            documents=self.docs, prompt=self.prompt, model_supports_sparse_attention=True
        )

        # Verify result
        assert "prompt" in result
        assert "attention_mask" in result
        assert result["prompt"] == self.prompt
        assert isinstance(result["attention_mask"], np.ndarray)

    def test_apply_gasa_with_block_diagonal(self) -> None:
        """Test applying GASA with block diagonal fallback."""
        # Create input IDs and attention mask
        input_ids = list(range(100))
        attention_mask = np.ones(100, dtype=np.int32)

        # Apply GASA
        result = self.gasa_service.apply_gasa(
            documents=self.docs,
            prompt=self.prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_supports_sparse_attention=False,
        )

        # Verify result
        assert "prompt" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["prompt"] == self.prompt
        assert isinstance(result["input_ids"], list)
        assert isinstance(result["attention_mask"], np.ndarray)

    @pytest.mark.skipif(True, reason="This test requires a real vLLM model and is slow")
    def test_gasa_with_vllm_model(self) -> None:
        """Test GASA with a real vLLM model."""
        try:
            # Create a model
            model = LLM.create(
                provider="vllm",
                model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                device="cpu",  # Use CPU for CI testing
            )

            # Apply GASA to the prompt
            result = self.gasa_service.apply_gasa(
                documents=self.docs, prompt=self.prompt, model_supports_sparse_attention=True
            )

            # Generate text with the modified prompt and attention mask
            response = asyncio.run(
                model.generate(
                    prompt=result["prompt"],
                    attention_mask=result.get("attention_mask"),
                    max_tokens=100,
                )
            )

            # Verify response
            assert response.text
            assert len(response.text) > 0
        except (ImportError, RuntimeError, ValueError) as e:
            pytest.skip(f"vLLM model loading failed: {e}")
