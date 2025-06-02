from __future__ import annotations

"""
Vector utilities for Saplings.

This module provides utility functions for vector operations.
"""

from typing import List

import numpy as np


def normalize_vectors(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize vectors to unit length.

    Args:
    ----
        vectors: List of vectors to normalize

    Returns:
    -------
        List[np.ndarray]: Normalized vectors

    """
    return [v / np.linalg.norm(v) for v in vectors]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
    ----
        a: First vector
        b: Second vector

    Returns:
    -------
        float: Cosine similarity

    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
    ----
        a: First vector
        b: Second vector

    Returns:
    -------
        float: Euclidean distance

    """
    return float(np.linalg.norm(a - b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate dot product between two vectors.

    Args:
    ----
        a: First vector
        b: Second vector

    Returns:
    -------
        float: Dot product

    """
    return float(np.dot(a, b))
