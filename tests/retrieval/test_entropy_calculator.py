"""
Tests for the entropy calculator module.
"""

import tempfile

from saplings.memory.document import Document, DocumentMetadata
from saplings.retrieval.config import EntropyConfig, RetrievalConfig
from saplings.retrieval.entropy_calculator import EntropyCalculator


class TestEntropyCalculator:
    """Tests for the EntropyCalculator class."""

    def setup_method(self):
        """Set up test environment."""
        # Create test documents
        self.docs = [
            Document(
                id="doc1",
                content="This is a document about machine learning and artificial intelligence.",
                metadata=DocumentMetadata(source="test1.txt"),
            ),
            Document(
                id="doc2",
                content="Python is a popular programming language for data science and machine learning.",
                metadata=DocumentMetadata(source="test2.txt"),
            ),
            Document(
                id="doc3",
                content="Natural language processing is a subfield of artificial intelligence.",
                metadata=DocumentMetadata(source="test3.txt"),
            ),
            Document(
                id="doc4",
                content="Deep learning is a subset of machine learning that uses neural networks.",
                metadata=DocumentMetadata(source="test4.txt"),
            ),
        ]

        # Create the entropy calculator
        self.config = EntropyConfig(
            threshold=0.1,
            max_iterations=3,
            min_documents=2,
            max_documents=10,
            window_size=3,
        )
        self.calculator = EntropyCalculator(self.config)

    def test_init_with_retrieval_config(self):
        """Test initialization with RetrievalConfig."""
        retrieval_config = RetrievalConfig(entropy=EntropyConfig(threshold=0.05))
        calculator = EntropyCalculator(retrieval_config)

        assert calculator.config.threshold == 0.05

    def test_calculate_entropy(self):
        """Test calculating entropy."""
        # Calculate entropy for a set of documents
        entropy = self.calculator.calculate_entropy(self.docs)

        # Entropy should be positive
        assert entropy > 0.0

        # Entropy should be different for different document sets
        entropy_subset = self.calculator.calculate_entropy(self.docs[:2])
        assert entropy != entropy_subset

    def test_entropy_with_diverse_vs_similar_documents(self):
        """Test entropy calculation with diverse vs similar document sets."""
        # Create a set of diverse documents
        diverse_docs = [
            Document(
                id="diverse1",
                content="Machine learning is a field of artificial intelligence focused on algorithms.",
                metadata=DocumentMetadata(source="diverse1.txt"),
            ),
            Document(
                id="diverse2",
                content="Quantum computing uses quantum mechanics to process information.",
                metadata=DocumentMetadata(source="diverse2.txt"),
            ),
            Document(
                id="diverse3",
                content="Climate change is affecting global weather patterns and ecosystems.",
                metadata=DocumentMetadata(source="diverse3.txt"),
            ),
        ]

        # Create a set of similar documents
        similar_docs = [
            Document(
                id="similar1",
                content="Machine learning algorithms can be supervised or unsupervised.",
                metadata=DocumentMetadata(source="similar1.txt"),
            ),
            Document(
                id="similar2",
                content="Deep learning is a subset of machine learning using neural networks.",
                metadata=DocumentMetadata(source="similar2.txt"),
            ),
            Document(
                id="similar3",
                content="Reinforcement learning is a type of machine learning algorithm.",
                metadata=DocumentMetadata(source="similar3.txt"),
            ),
        ]

        # Calculate entropy for both sets
        diverse_entropy = self.calculator.calculate_entropy(diverse_docs)
        similar_entropy = self.calculator.calculate_entropy(similar_docs)

        # Diverse documents should have higher entropy
        assert diverse_entropy > similar_entropy

        # Test normalized entropy
        self.calculator.config.use_normalized_entropy = True

        diverse_norm_entropy = self.calculator.calculate_entropy(diverse_docs)
        similar_norm_entropy = self.calculator.calculate_entropy(similar_docs)

        # Normalized entropy should be between 0 and 1
        assert 0 <= diverse_norm_entropy <= 1
        assert 0 <= similar_norm_entropy <= 1

        # Diverse documents should still have higher normalized entropy
        assert diverse_norm_entropy > similar_norm_entropy

    def test_calculate_entropy_empty_documents(self):
        """Test calculating entropy for empty documents."""
        entropy = self.calculator.calculate_entropy([])

        assert entropy == 0.0

    def test_calculate_entropy_change(self):
        """Test calculating entropy change."""
        # Calculate initial entropy
        change1 = self.calculator.calculate_entropy_change(self.docs[:2])

        # Should be equal to the entropy of the first set
        assert change1 > 0.0

        # Calculate entropy change with more documents
        change2 = self.calculator.calculate_entropy_change(self.docs[:3])

        # Should be non-zero (entropy changes with different document sets)
        assert change2 != 0.0

        # Calculate entropy change with same documents
        change3 = self.calculator.calculate_entropy_change(self.docs[:3])

        # Should be close to zero (no change in entropy)
        assert abs(change3) < 0.01

    def test_should_terminate_min_documents(self):
        """Test termination based on minimum documents."""
        # Not enough documents
        result = self.calculator.should_terminate([self.docs[0]], 1)

        assert result is False

    def test_should_terminate_max_documents(self):
        """Test termination based on maximum documents."""
        # Configure with low max_documents
        self.calculator.config.max_documents = 3

        # Too many documents
        result = self.calculator.should_terminate(self.docs, 1)

        assert result is True

    def test_should_terminate_max_iterations(self):
        """Test termination based on maximum iterations."""
        # Configure with low max_iterations
        self.calculator.config.max_iterations = 2

        # Too many iterations
        result = self.calculator.should_terminate(self.docs[:2], 2)

        assert result is True

    def test_should_terminate_entropy_threshold(self):
        """Test termination based on entropy threshold."""
        # Calculate initial entropy
        self.calculator.calculate_entropy_change(self.docs[:2])

        # Add similar documents to keep entropy change low
        similar_docs = [
            Document(
                id="doc5",
                content="Machine learning is a field of artificial intelligence.",
                metadata=DocumentMetadata(source="test5.txt"),
            ),
            Document(
                id="doc6",
                content="AI and machine learning are related fields.",
                metadata=DocumentMetadata(source="test6.txt"),
            ),
        ]

        # Set a high threshold
        self.calculator.config.threshold = 0.5

        # Calculate entropy change with similar documents
        self.calculator.calculate_entropy_change(self.docs[:2] + similar_docs[:1])

        # Should terminate due to small entropy change
        result = self.calculator.should_terminate(self.docs[:2] + similar_docs, 2)

        assert result is True

    def test_reset(self):
        """Test resetting the entropy calculator."""
        # Calculate some entropy values
        self.calculator.calculate_entropy_change(self.docs[:2])
        self.calculator.calculate_entropy_change(self.docs[:3])

        # Reset the calculator
        self.calculator.reset()

        # History should be empty
        assert len(self.calculator.entropy_history) == 0

    def test_save_and_load(self):
        """Test saving and loading the entropy calculator."""
        # Save the calculator
        with tempfile.TemporaryDirectory() as temp_dir:
            self.calculator.save(temp_dir)

            # Create a new calculator and load the saved data
            new_calculator = EntropyCalculator()
            new_calculator.load(temp_dir)

            # Check that the loaded calculator has the same configuration
            assert new_calculator.config.threshold == self.calculator.config.threshold
            assert new_calculator.config.max_iterations == self.calculator.config.max_iterations
            assert new_calculator.config.min_documents == self.calculator.config.min_documents
            assert new_calculator.config.max_documents == self.calculator.config.max_documents
            assert new_calculator.config.window_size == self.calculator.config.window_size
