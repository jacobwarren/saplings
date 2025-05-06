from __future__ import annotations

"""
Entropy calculator module for Saplings.

This module provides the entropy-based termination logic for the cascade retriever.
"""


import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from saplings.retrieval.config import EntropyConfig, RetrievalConfig

if TYPE_CHECKING:
    from saplings.memory.document import Document

logger = logging.getLogger(__name__)


class EntropyCalculator:
    """
    Entropy calculator for determining retrieval sufficiency.

    This class calculates the information entropy of retrieval results
    to determine when sufficient information has been retrieved.
    """

    def __init__(
        self,
        config: RetrievalConfig | EntropyConfig | None = None,
    ) -> None:
        """
        Initialize the entropy calculator.

        Args:
        ----
            config: Retrieval or entropy configuration

        """
        # Extract entropy config from RetrievalConfig if needed
        if config is None:
            self.config = EntropyConfig(
                threshold=0.1,
                max_iterations=3,
                min_documents=5,
                max_documents=50,
                use_normalized_entropy=True,
                window_size=3,
            )
        elif isinstance(config, RetrievalConfig):
            self.config = config.entropy
        else:
            self.config = config

        # Initialize count vectorizer for term frequency analysis
        self.vectorizer = CountVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 1),
        )

        # Initialize history for tracking entropy changes
        self.entropy_history: list[float] = []

    def calculate_entropy(self, documents: list[Document]) -> float:
        """
        Calculate the information entropy of a set of documents.

        Args:
        ----
            documents: Documents to calculate entropy for

        Returns:
        -------
            float: Entropy value

        """
        if not documents:
            return 0.0

        # Extract document contents
        contents = [doc.content for doc in documents]

        # Calculate term frequencies
        try:
            # Get term counts
            term_counts = self.vectorizer.fit_transform(contents)

            # Convert to dense array for term frequencies
            # Handle different matrix types
            if sp.issparse(term_counts):
                # For sparse matrices (what CountVectorizer typically returns)
                term_counts_array = term_counts.toarray()  # type: ignore
            else:
                # If it's already a dense array
                term_counts_array = np.array(term_counts)

            term_freqs = np.sum(term_counts_array, axis=0)

            # Calculate probability distribution
            total_terms = term_freqs.sum()
            # Index for total terms
            TOTAL_TERMS_INDEX = 0
            if total_terms == TOTAL_TERMS_INDEX:
                return 0.0

            probabilities = term_freqs / total_terms

            # Filter out zero probabilities
            probabilities = probabilities[probabilities > 0]

            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))

            # Normalize if configured
            if self.config.use_normalized_entropy:
                max_entropy = math.log2(len(probabilities))
                # Threshold for max entropy
                MAX_ENTROPY_THRESHOLD = 0
                if max_entropy > MAX_ENTROPY_THRESHOLD:
                    entropy /= max_entropy

            return float(entropy)

        except Exception as e:
            logger.exception(f"Error calculating entropy: {e}")
            return 0.0

    def calculate_entropy_change(self, documents: list[Document]) -> float:
        """
        Calculate the change in entropy compared to previous iterations.

        Args:
        ----
            documents: Current set of documents

        Returns:
        -------
            float: Entropy change (negative means decreasing entropy)

        """
        current_entropy = self.calculate_entropy(documents)

        if not self.entropy_history:
            self.entropy_history.append(current_entropy)
            return current_entropy

        # Calculate change from previous entropy
        previous_entropy = self.entropy_history[-1]
        entropy_change = current_entropy - previous_entropy

        # Update history
        self.entropy_history.append(current_entropy)

        # Limit history to window size
        if len(self.entropy_history) > self.config.window_size:
            self.entropy_history.pop(0)

        return entropy_change

    def should_terminate(self, documents: list[Document], iteration: int) -> bool:
        """
        Determine if the retrieval process should terminate.

        Args:
        ----
            documents: Current set of documents
            iteration: Current iteration number

        Returns:
        -------
            bool: True if retrieval should terminate, False otherwise

        """
        # Check minimum number of documents
        if len(documents) < self.config.min_documents:
            return False

        # Check maximum number of documents
        if len(documents) >= self.config.max_documents:
            return True

        # Check maximum iterations
        if iteration >= self.config.max_iterations:
            return True

        # Calculate entropy change
        entropy_change = self.calculate_entropy_change(documents)

        # Check if entropy change is below threshold
        return abs(entropy_change) < self.config.threshold

    def reset(self):
        """Reset the entropy calculator state."""
        self.entropy_history.clear()

    def save(self, directory: str) -> None:
        """
        Save the entropy calculator configuration to disk.

        Args:
        ----
            directory: Directory to save to

        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(directory_path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f)

        logger.info(f"Saved entropy calculator configuration to {directory}")

    def load(self, directory: str) -> None:
        """
        Load the entropy calculator configuration from disk.

        Args:
        ----
            directory: Directory to load from

        """
        directory_path = Path(directory)

        # Load config
        config_path = directory_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = EntropyConfig(**config_data)

        logger.info(f"Loaded entropy calculator configuration from {directory}")
