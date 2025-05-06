from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from saplings.gasa.config import FallbackStrategy, GASAConfig
from saplings.gasa.core.types import MaskFormat, MaskType
from saplings.gasa.service.gasa_service import GASAService
from saplings.memory import DependencyGraph, Document

"""
Unit tests for the GASA service.
"""


class TestGASAService:
    """Test the GASA service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create a simple graph
        self.graph = DependencyGraph()

        # Create documents
        self.doc1 = Document(id="doc1", content="Document 1 content")
        self.doc2 = Document(id="doc2", content="Document 2 content")
        self.doc3 = Document(id="doc3", content="Document 3 content")

        # Add documents to graph
        self.graph.add_document_node(self.doc1)
        self.graph.add_document_node(self.doc2)
        self.graph.add_document_node(self.doc3)

        # Add relationships
        self.graph.add_relationship("doc1", "doc2", "relates_to", 0.9)
        self.graph.add_relationship("doc2", "doc3", "relates_to", 0.8)

        # Create config for testing
        self.config = GASAConfig.default()
        # Override settings for testing
        self.config.max_hops = 2
        self.config.global_tokens = []
        self.config.summary_token = ""
        self.config.add_summary_token = False
        self.config.block_size = 0
        self.config.overlap = 0
        self.config.cache_masks = False
        self.config.enable_shadow_model = False
        self.config.enable_prompt_composer = False
        self.config.focus_tags = False
        self.config.core_tag = ""
        self.config.near_tag = ""
        self.config.summary_tag = ""

        # Create GASA service with patch
        with (
            patch("saplings.gasa.service.gasa_service.BlockDiagonalPacker") as mock_packer_class,
            patch("saplings.gasa.service.gasa_service.PromptComposer") as mock_composer_class,
        ):
            # Setup mock packer
            mock_packer = MagicMock()
            mock_packer.reorder_tokens.return_value = {
                "input_ids": list(range(30)),
                "attention_mask": np.ones((30, 30), dtype=np.int32),
                "reordering": list(range(30)),
            }
            mock_packer_class.return_value = mock_packer

            # Setup mock composer
            mock_composer = MagicMock()
            mock_composer.compose_prompt.return_value = "Composed prompt"
            mock_composer_class.return_value = mock_composer

            # Create service
            self.service = GASAService(graph=self.graph, config=self.config)

            # Store mocks for later use
            self.mock_packer = mock_packer
            self.mock_composer = mock_composer

        # Mock the mask builder
        self.mock_mask = np.ones((30, 30), dtype=np.int32)
        self.service.mask_builder.build_mask = MagicMock(return_value=self.mock_mask)

    def test_initialization(self) -> None:
        """Test GASA service initialization."""
        assert self.service.graph is self.graph
        assert self.service.config is self.config
        assert self.service.mask_builder is not None
        # Access to packer and prompt_composer is through private attributes
        assert hasattr(self.service, "_packer")
        assert hasattr(self.service, "_prompt_composer")

    def test_build_mask(self) -> None:
        """Test building a mask."""
        # Create a prompt using the documents
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Build mask
        mask = self.service.build_mask(
            documents=[self.doc1, self.doc2, self.doc3],
            prompt=prompt,
            format=MaskFormat.DENSE,
            mask_type=MaskType.ATTENTION,
        )

        # Verify mask
        assert mask is self.mock_mask
        self.service.mask_builder.build_mask.assert_called_once_with(
            documents=[self.doc1, self.doc2, self.doc3],
            prompt=prompt,
            format=MaskFormat.DENSE,
            mask_type=MaskType.ATTENTION,
        )

    def test_apply_gasa_with_sparse_attention(self) -> None:
        """Test applying GASA with sparse attention support."""
        # Create a prompt using the documents
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Apply GASA
        result = self.service.apply_gasa(
            documents=[self.doc1, self.doc2, self.doc3],
            prompt=prompt,
            model_supports_sparse_attention=True,
        )

        # Verify result
        assert result["prompt"] == prompt
        assert result["attention_mask"] is self.mock_mask
        self.service.mask_builder.build_mask.assert_called_once()

    def test_apply_gasa_with_block_diagonal(self) -> None:
        """Test applying GASA with block diagonal fallback."""
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Reset mock to ensure clean state
        self.mock_packer.reset_mock()

        # Set up return value
        self.mock_packer.reorder_tokens.return_value = {
            "input_ids": list(range(30)),
            "attention_mask": np.ones((30, 30), dtype=np.int32),
            "reordering": list(range(30)),
        }

        # Call the method
        result = self.service.apply_gasa(
            documents=[self.doc1, self.doc2, self.doc3],
            prompt=prompt,
            input_ids=list(range(30)),
            attention_mask=np.ones((30,), dtype=np.int32),
            model_supports_sparse_attention=False,
        )

        # Verify results
        assert result["input_ids"] == list(range(30))
        assert np.array_equal(result["attention_mask"], np.ones((30, 30), dtype=np.int32))
        self.mock_packer.reorder_tokens.assert_called_once()

    def test_apply_gasa_with_prompt_composer(self) -> None:
        """Test applying GASA with prompt composer fallback."""
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Set fallback strategy to prompt composer
        self.service.config.fallback_strategy = FallbackStrategy.PROMPT_COMPOSER

        # Reset mock to ensure clean state
        self.mock_composer.reset_mock()

        # Set up return value
        self.mock_composer.compose_prompt.return_value = "Composed prompt"

        # Call the method
        result = self.service.apply_gasa(
            documents=[self.doc1, self.doc2, self.doc3],
            prompt=prompt,
            model_supports_sparse_attention=False,
            system_prompt="System prompt",
        )

        # Verify results
        assert result["prompt"] == "Composed prompt"
        self.mock_composer.compose_prompt.assert_called_once_with(
            documents=[self.doc1, self.doc2, self.doc3],
            prompt=prompt,
            system_prompt="System prompt",
        )

    def test_apply_gasa_disabled(self) -> None:
        """Test applying GASA when disabled."""
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Disable GASA
        self.service.config.enabled = False

        # Reset mocks to ensure clean state
        self.service.mask_builder.build_mask.reset_mock()
        self.mock_packer.reset_mock()
        self.mock_composer.reset_mock()

        # Call the method
        result = self.service.apply_gasa(
            documents=[self.doc1, self.doc2, self.doc3],
            prompt=prompt,
            input_ids=list(range(30)),
            attention_mask=np.ones((30,), dtype=np.int32),
            model_supports_sparse_attention=True,
        )

        # Verify results
        assert result["prompt"] == prompt
        assert result["input_ids"] == list(range(30))
        assert np.array_equal(result["attention_mask"], np.ones((30,), dtype=np.int32))

        # Verify mocks were not called
        self.service.mask_builder.build_mask.assert_not_called()
        self.mock_packer.reorder_tokens.assert_not_called()
        self.mock_composer.compose_prompt.assert_not_called()
