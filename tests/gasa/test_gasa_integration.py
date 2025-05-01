"""
Tests for GASA integration with the Executor.
"""

import asyncio
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig, MaskBuilder, MaskFormat, MaskType
from saplings.memory.document import Document
from saplings.memory.graph import DependencyGraph, DocumentNode


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        """Initialize the mock tokenizer."""
        self.unk_token_id = 100
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.mask_token_id = 103

        # Add special tokens
        self.special_tokens = {
            "[CLS]": self.cls_token_id,
            "[SEP]": self.sep_token_id,
            "[MASK]": self.mask_token_id,
            "[PAD]": self.pad_token_id,
            "<s>": 1,
            "</s>": 2,
            "[SUM]": 3,
        }

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        """Tokenize text."""
        # Simple mock implementation that just counts words
        tokens = text.split()

        # Add special tokens if requested
        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # Convert to token IDs
        input_ids = [self.convert_tokens_to_ids(token) for token in tokens]

        # Create attention mask (1 for all tokens)
        attention_mask = [1] * len(input_ids)

        if return_tensors == "pt":
            # Return a structure similar to what PyTorch tokenizers return
            return MockTokenizerOutput(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def convert_tokens_to_ids(self, token):
        """Convert token to ID."""
        # Check if it's a special token
        if token in self.special_tokens:
            return self.special_tokens[token]

        # Simple hash for regular tokens to ensure uniqueness
        return hash(token) % 10000 + 1000

    def convert_ids_to_tokens(self, ids):
        """Convert IDs to tokens."""
        if isinstance(ids, int):
            # Check if it's a special token ID
            for token, token_id in self.special_tokens.items():
                if token_id == ids:
                    return token
            return f"token_{ids}"

        return [self.convert_ids_to_tokens(id) for id in ids]

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            token = self.convert_ids_to_tokens(token_id)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def batch_decode(self, sequences, skip_special_tokens=False):
        """Decode multiple sequences."""
        return [self.decode(seq, skip_special_tokens) for seq in sequences]

    def get_vocab(self):
        """Get the vocabulary."""
        return {token: id for token, id in self.special_tokens.items()}


class MockTokenizerOutput:
    """Mock tokenizer output for testing."""

    def __init__(self, input_ids, attention_mask=None):
        """Initialize the mock tokenizer output."""
        self.input_ids = [input_ids]
        self.attention_mask = (
            [attention_mask] if attention_mask is not None else [[1] * len(input_ids)]
        )


class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, model_uri, **kwargs):
        """Initialize the mock LLM."""
        self.model_uri = model_uri
        self.kwargs = kwargs
        self.tokenizer = MockTokenizer()
        self.generate_calls = []
        self.attention_masks = []

    async def generate(
        self, prompt, max_tokens=None, temperature=None, attention_mask=None, **kwargs
    ) -> LLMResponse:
        """Generate text from the model."""
        # Record the call
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "attention_mask": attention_mask,
                "kwargs": kwargs,
            }
        )

        # Store the attention mask for testing
        if attention_mask is not None:
            self.attention_masks.append(attention_mask)

        # Return a mock response
        return LLMResponse(
            text=f"Response to: {prompt[:50]}...",
            model_uri=str(self.model_uri),
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 20,
                "total_tokens": len(prompt.split()) + 20,
            },
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def generate_streaming(
        self, prompt, max_tokens=None, temperature=None, attention_mask=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text from the model with streaming output."""
        # Record the call
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "attention_mask": attention_mask,
                "kwargs": kwargs,
                "streaming": True,
            }
        )

        # Store the attention mask for testing
        if attention_mask is not None:
            self.attention_masks.append(attention_mask)

        # Yield a mock response in chunks
        response = f"Response to: {prompt[:50]}..."
        chunks = [response[i : i + 5] for i in range(0, len(response), 5)]

        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)

    def get_metadata(self) -> ModelMetadata:
        """Get metadata about the model."""
        return ModelMetadata(
            name="mock-model",
            provider="mock-provider",
            version="latest",
            roles=[ModelRole.EXECUTOR, ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=2048,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        return len(text.split())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of a request."""
        return (prompt_tokens + completion_tokens) * 0.0001


class TestGASAIntegration:
    """Tests for GASA integration with the Executor."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockLLM("mock://model/latest")

    @pytest.fixture
    def mock_dependency_graph(self):
        """Create a mock dependency graph for testing."""
        graph = DependencyGraph()

        # Add some documents to the graph
        doc1 = Document(
            id="doc1",
            content="This is document 1.",
            metadata={"source": "test"},
        )
        doc2 = Document(
            id="doc2",
            content="This is document 2.",
            metadata={"source": "test"},
        )

        # Add document nodes
        graph.add_document_node(doc1)
        graph.add_document_node(doc2)

        # Add an edge between the documents
        graph.add_edge(
            source_id="doc1",
            target_id="doc2",
            relationship_type="related_to",
            weight=0.8,
        )

        return graph

    @pytest.fixture
    def documents(self):
        """Create test documents."""
        return [
            Document(
                id="doc1",
                content="This is document 1.",
                metadata={"source": "test"},
            ),
            Document(
                id="doc2",
                content="This is document 2.",
                metadata={"source": "test"},
            ),
        ]

    @pytest.fixture
    def mock_mask_builder(self, mock_dependency_graph):
        """Create a mock MaskBuilder for testing."""

        class MockMaskBuilder(MaskBuilder):
            def __init__(self, graph, config=None, tokenizer=None):
                self.graph = graph
                self.config = config or GASAConfig()
                self.tokenizer = tokenizer
                self.build_count = 0

            def build_mask(
                self, documents, prompt, format=MaskFormat.DENSE, mask_type=MaskType.ATTENTION
            ):
                """Build a mock attention mask."""
                self.build_count += 1

                # Create a simple square mask of ones
                seq_len = len(prompt.split()) + 10  # Add some padding
                mask = np.ones((seq_len, seq_len), dtype=np.float32)

                # Add some structure to the mask based on documents
                if documents:
                    # Create some zeros in the mask to simulate sparse attention
                    for i in range(0, seq_len, 3):
                        for j in range(0, seq_len, 3):
                            if i != j and (i % 2 == 0 or j % 2 == 0):
                                mask[i, j] = 0

                return mask

        return MockMaskBuilder(graph=mock_dependency_graph)

    @pytest.fixture
    def executor_with_gasa(self, mock_llm, mock_dependency_graph, mock_mask_builder):
        """Create an executor with GASA enabled."""
        config = ExecutorConfig(
            enable_gasa=True,
        )
        gasa_config = GASAConfig(
            enabled=True,
            max_hops=2,
            mask_strategy="binary",
            cache_masks=True,
        )

        executor = Executor(
            model=mock_llm,
            config=config,
            gasa_config=gasa_config,
            dependency_graph=mock_dependency_graph,
        )

        # Replace the mask builder with our mock
        executor.mask_builder = mock_mask_builder

        return executor

    @pytest.fixture
    def executor_without_gasa(self, mock_llm):
        """Create an executor with GASA disabled."""
        config = ExecutorConfig(
            enable_gasa=False,
        )

        return Executor(
            model=mock_llm,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_gasa_initialization(self, executor_with_gasa, executor_without_gasa):
        """Test GASA initialization in the Executor."""
        # Check that GASA is initialized when enabled
        assert executor_with_gasa.mask_builder is not None
        assert isinstance(executor_with_gasa.mask_builder, MaskBuilder)

        # Check that GASA is not initialized when disabled
        assert executor_without_gasa.mask_builder is None

    @pytest.mark.asyncio
    async def test_gasa_mask_injection_in_draft(self, executor_with_gasa, documents):
        """Test GASA mask injection in draft generation."""
        # Execute with documents
        prompt = "Summarize these documents:"
        await executor_with_gasa.execute(prompt=prompt, documents=documents)

        # Check that an attention mask was passed to the model
        assert len(executor_with_gasa.model.attention_masks) > 0

        # Check the first attention mask
        mask = executor_with_gasa.model.attention_masks[0]
        assert isinstance(mask, np.ndarray)

        # The mask should be a square matrix
        assert mask.ndim == 2
        assert mask.shape[0] == mask.shape[1]

    @pytest.mark.asyncio
    async def test_gasa_mask_injection_in_streaming(self, executor_with_gasa, documents):
        """Test GASA mask injection in streaming generation."""
        # Execute with documents and streaming
        prompt = "Summarize these documents:"
        await executor_with_gasa.execute(
            prompt=prompt,
            documents=documents,
            stream=True,
        )

        # Check that attention masks were passed to the model
        assert len(executor_with_gasa.model.attention_masks) > 0

        # Check the attention masks
        for mask in executor_with_gasa.model.attention_masks:
            assert isinstance(mask, np.ndarray)
            assert mask.ndim == 2
            assert mask.shape[0] == mask.shape[1]

    @pytest.mark.asyncio
    async def test_no_gasa_mask_without_documents(self, executor_with_gasa):
        """Test that no GASA mask is used when no documents are provided."""
        # Execute without documents
        prompt = "Generate a story:"
        await executor_with_gasa.execute(prompt=prompt)

        # Check that no attention mask was passed to the model
        assert len(executor_with_gasa.model.attention_masks) == 0

    @pytest.mark.asyncio
    async def test_no_gasa_mask_when_disabled(self, executor_without_gasa, documents):
        """Test that no GASA mask is used when GASA is disabled."""
        # Execute with documents but GASA disabled
        prompt = "Summarize these documents:"
        await executor_without_gasa.execute(prompt=prompt, documents=documents)

        # Check that no attention mask was passed to the model
        assert len(executor_without_gasa.model.attention_masks) == 0

    @pytest.mark.asyncio
    async def test_gasa_mask_format(self, executor_with_gasa, documents):
        """Test the format of the GASA mask."""
        # Execute with documents
        prompt = "Summarize these documents:"
        await executor_with_gasa.execute(prompt=prompt, documents=documents)

        # Check the mask format
        mask = executor_with_gasa.model.attention_masks[0]

        # The mask should be binary (0s and 1s)
        assert np.all(np.logical_or(mask == 0, mask == 1))

        # The mask should have 1s on the diagonal (self-attention)
        assert np.all(np.diag(mask) == 1)

    @pytest.mark.asyncio
    async def test_gasa_mask_caching(self, executor_with_gasa, documents):
        """Test that GASA masks are cached."""
        # Execute with the same prompt and documents twice
        prompt = "Summarize these documents:"
        await executor_with_gasa.execute(prompt=prompt, documents=documents)

        # Get the number of masks built
        mask_count_before = len(executor_with_gasa.model.attention_masks)

        # Execute again with the same prompt and documents
        await executor_with_gasa.execute(prompt=prompt, documents=documents)

        # Get the number of masks built after
        mask_count_after = len(executor_with_gasa.model.attention_masks)

        # The number of masks should have increased by the same amount
        # (one for draft, one for final)
        assert mask_count_after - mask_count_before == 2

        # The masks should be the same
        assert np.array_equal(
            executor_with_gasa.model.attention_masks[0],
            executor_with_gasa.model.attention_masks[2],
        )
        assert np.array_equal(
            executor_with_gasa.model.attention_masks[1],
            executor_with_gasa.model.attention_masks[3],
        )

    @pytest.mark.asyncio
    async def test_gasa_with_different_mask_format(
        self, mock_llm, mock_dependency_graph, documents
    ):
        """Test GASA with different mask formats."""
        # Create an executor with GASA configured to use sparse format
        config = ExecutorConfig(enable_gasa=True)
        gasa_config = GASAConfig(
            enabled=True,
            max_hops=2,
            mask_strategy="binary",
            cache_masks=True,
        )

        # Create a custom mask builder that uses sparse format
        class CustomMaskBuilder:
            def __init__(self, graph, config=None, tokenizer=None):
                self.graph = graph
                self.config = config or GASAConfig()
                self.tokenizer = tokenizer
                self.build_count = 0

            def build_mask(
                self, documents, prompt, format=MaskFormat.SPARSE, mask_type=MaskType.ATTENTION
            ):
                """Build a mock attention mask."""
                self.build_count += 1

                # Create a simple square mask of ones
                seq_len = len(prompt.split()) + 10  # Add some padding

                # For sparse format, return a scipy sparse matrix
                if format == MaskFormat.SPARSE:
                    import scipy.sparse as sp

                    # Convert to dense first for simplicity
                    mask = np.ones((seq_len, seq_len), dtype=np.float32)
                    # Add some sparsity
                    for i in range(0, seq_len, 3):
                        for j in range(0, seq_len, 3):
                            if i != j and (i % 2 == 0 or j % 2 == 0):
                                mask[i, j] = 0
                    # Convert to sparse
                    return sp.csr_matrix(mask)

                # For dense format, return a numpy array
                mask = np.ones((seq_len, seq_len), dtype=np.float32)
                # Add some sparsity
                for i in range(0, seq_len, 3):
                    for j in range(0, seq_len, 3):
                        if i != j and (i % 2 == 0 or j % 2 == 0):
                            mask[i, j] = 0
                return mask

        # Create the executor with the custom mask builder
        executor = Executor(
            model=mock_llm,
            config=config,
            gasa_config=gasa_config,
            dependency_graph=mock_dependency_graph,
        )

        # Replace the mask builder with our custom one
        executor.mask_builder = CustomMaskBuilder(
            graph=mock_dependency_graph,
            config=gasa_config,
            tokenizer=mock_llm.tokenizer,
        )

        # Execute with documents
        prompt = "Summarize these documents:"
        await executor.execute(prompt=prompt, documents=documents)

        # Check that an attention mask was passed to the model
        assert len(mock_llm.attention_masks) > 0
