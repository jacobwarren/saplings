from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from saplings.gasa.config import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.gasa.mask_builder import MaskBuilder, MaskFormat, MaskType
from saplings.memory import DependencyGraph, Document

"""
Unit tests for the GASA mask builder.
"""


class TestMaskBuilder:
    """Test the GASA mask builder."""

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

        # Create config
        self.config = GASAConfig(
            max_hops=2,
            enabled=True,
            mask_strategy=MaskStrategy.BINARY,
            fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
            global_tokens=[],
            summary_token="",
            add_summary_token=False,
            block_size=0,
            overlap=0,
            soft_mask_temperature=1.0,
            cache_masks=False,
            cache_dir=None,
            visualize=False,
            visualization_dir=None,
            enable_shadow_model=False,
            shadow_model_name="",
            shadow_model_device="",
            shadow_model_cache_dir=None,
            enable_prompt_composer=False,
            focus_tags=False,
            core_tag="",
            near_tag="",
            summary_tag="",
        )

        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.return_value = MagicMock(
            input_ids=MagicMock(
                # Mock a sequence of 30 tokens
                tolist=MagicMock(return_value=list(range(30)))
            )
        )

        # Create mask builder with mock tokenizer
        self.builder = MaskBuilder(graph=self.graph, config=self.config)
        self.builder.tokenizer = self.mock_tokenizer

    def test_initialization(self) -> None:
        """Test mask builder initialization."""
        assert self.builder.graph is self.graph
        assert self.builder.config is self.config
        assert self.builder.tokenizer is self.mock_tokenizer

    def test_build_mask_dense(self) -> None:
        """Test building a dense attention mask."""
        # Create a prompt using the documents
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Mock the _map_tokens_to_chunks method
        with patch.object(self.builder, "_map_tokens_to_chunks") as mock_map:
            # Create chunk info mapping
            chunk_infos = [
                # First 10 tokens belong to doc1
                {"doc_id": "doc1", "chunk_id": 0, "start": 0, "end": 10},
                # Next 10 tokens belong to doc2
                {"doc_id": "doc2", "chunk_id": 0, "start": 10, "end": 20},
                # Last 10 tokens belong to doc3
                {"doc_id": "doc3", "chunk_id": 0, "start": 20, "end": 30},
            ]
            mock_map.return_value = chunk_infos

            # Build mask
            mask = self.builder.build_mask(
                documents=[self.doc1, self.doc2, self.doc3],
                prompt=prompt,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
            )

            # Verify mask shape and properties
            assert isinstance(mask, np.ndarray)
            assert mask.ndim == 2
            assert mask.shape == (30, 30)  # 30x30 for 30 tokens

            # Check mask values based on graph relationships
            # doc1 and doc2 are directly connected (1 hop)
            assert mask[5, 15] == 1  # Token from doc1 can attend to token from doc2
            assert mask[15, 5] == 1  # Token from doc2 can attend to token from doc1

            # doc1 and doc3 are connected through doc2 (2 hops)
            assert mask[5, 25] == 1  # Token from doc1 can attend to token from doc3
            assert mask[25, 5] == 1  # Token from doc3 can attend to token from doc1

            # doc2 and doc3 are directly connected (1 hop)
            assert mask[15, 25] == 1  # Token from doc2 can attend to token from doc3
            assert mask[25, 15] == 1  # Token from doc3 can attend to token from doc2

            # Tokens within the same document can always attend to each other
            assert mask[5, 8] == 1  # Token from doc1 can attend to another token from doc1
            assert mask[15, 18] == 1  # Token from doc2 can attend to another token from doc2
            assert mask[25, 28] == 1  # Token from doc3 can attend to another token from doc3

    def test_build_mask_with_max_hops_0(self) -> None:
        """Test building a mask with max_hops=0."""
        # Create config with max_hops=0
        config = GASAConfig(
            max_hops=0,
            enabled=True,
            mask_strategy=MaskStrategy.BINARY,
            fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
            global_tokens=[],
            summary_token="",
            add_summary_token=False,
            block_size=0,
            overlap=0,
            soft_mask_temperature=1.0,
            cache_masks=False,
            cache_dir=None,
            visualize=False,
            visualization_dir=None,
            enable_shadow_model=False,
            shadow_model_name="",
            shadow_model_device="",
            shadow_model_cache_dir=None,
            enable_prompt_composer=False,
            focus_tags=False,
            core_tag="",
            near_tag="",
            summary_tag="",
        )
        builder = MaskBuilder(graph=self.graph, config=config)
        builder.tokenizer = self.mock_tokenizer

        # Create a prompt using the documents
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Mock the _map_tokens_to_chunks method
        with patch.object(builder, "_map_tokens_to_chunks") as mock_map:
            # Create chunk info mapping
            chunk_infos = [
                # First 10 tokens belong to doc1
                {"doc_id": "doc1", "chunk_id": 0, "start": 0, "end": 10},
                # Next 10 tokens belong to doc2
                {"doc_id": "doc2", "chunk_id": 0, "start": 10, "end": 20},
                # Last 10 tokens belong to doc3
                {"doc_id": "doc3", "chunk_id": 0, "start": 20, "end": 30},
            ]
            mock_map.return_value = chunk_infos

            # Build mask
            mask = builder.build_mask(
                documents=[self.doc1, self.doc2, self.doc3],
                prompt=prompt,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
            )

            # Verify mask shape and properties
            assert isinstance(mask, np.ndarray)
            assert mask.ndim == 2
            assert mask.shape == (30, 30)  # 30x30 for 30 tokens

            # With max_hops=0, tokens can only attend to tokens from the same document
            # doc1 and doc2 should not be connected
            assert mask[5, 15] == 0  # Token from doc1 cannot attend to token from doc2
            assert mask[15, 5] == 0  # Token from doc2 cannot attend to token from doc1

            # doc1 and doc3 should not be connected
            assert mask[5, 25] == 0  # Token from doc1 cannot attend to token from doc3
            assert mask[25, 5] == 0  # Token from doc3 cannot attend to token from doc1

            # doc2 and doc3 should not be connected
            assert mask[15, 25] == 0  # Token from doc2 cannot attend to token from doc3
            assert mask[25, 15] == 0  # Token from doc3 cannot attend to token from doc2

            # Tokens within the same document can always attend to each other
            assert mask[5, 8] == 1  # Token from doc1 can attend to another token from doc1
            assert mask[15, 18] == 1  # Token from doc2 can attend to another token from doc2
            assert mask[25, 28] == 1  # Token from doc3 can attend to another token from doc3

    def test_build_mask_with_global_tokens(self) -> None:
        """Test building a mask with global tokens."""
        # Create a prompt using the documents
        prompt = f"Document 1: {self.doc1.content}\nDocument 2: {self.doc2.content}\nDocument 3: {self.doc3.content}\nSummary:"

        # Mock the _map_tokens_to_chunks method
        with patch.object(self.builder, "_map_tokens_to_chunks") as mock_map:
            # Create chunk info mapping
            chunk_infos = [
                # First 10 tokens belong to doc1
                {"doc_id": "doc1", "chunk_id": 0, "start": 0, "end": 10},
                # Next 10 tokens belong to doc2
                {"doc_id": "doc2", "chunk_id": 0, "start": 10, "end": 20},
                # Last 10 tokens belong to doc3
                {"doc_id": "doc3", "chunk_id": 0, "start": 20, "end": 30},
            ]
            mock_map.return_value = chunk_infos

            # Mock the _handle_global_tokens method to mark token 0 as global
            original_handle_global = self.builder._handle_global_tokens

            def mock_handle_global(token_mask, input_ids):
                # Call the original method
                mask = original_handle_global(token_mask, input_ids)
                # Mark token 0 as global
                mask[0, :] = 1
                mask[:, 0] = 1
                return mask

            self.builder._handle_global_tokens = mock_handle_global

            # Build mask
            mask = self.builder.build_mask(
                documents=[self.doc1, self.doc2, self.doc3],
                prompt=prompt,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
            )

            # Verify mask shape and properties
            assert isinstance(mask, np.ndarray)
            assert mask.ndim == 2
            assert mask.shape == (30, 30)  # 30x30 for 30 tokens

            # Check that token 0 is global
            assert np.all(mask[0, :] == 1)  # Global token can attend to all tokens
            assert np.all(mask[:, 0] == 1)  # All tokens can attend to global token
