"""
Tests for the MemoryConfig class in the public API.

These tests verify that the MemoryConfig class correctly configures memory components.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from saplings.api.memory import MemoryConfig


def test_memory_config_basic():
    """Test creating a basic memory configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = str(Path(temp_dir) / "memory")
        config = MemoryConfig(memory_path=memory_path)

        # Verify default settings
        assert config.memory_path == memory_path
        assert config.indexer_type == "simple"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.enable_embeddings is True
        assert config.enable_dependency_graph is True


def test_memory_config_minimal():
    """Test creating a minimal memory configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = str(Path(temp_dir) / "memory")
        config = MemoryConfig.minimal(memory_path=memory_path)

        # Verify minimal settings
        assert config.memory_path == memory_path
        assert config.enable_embeddings is False
        assert config.enable_dependency_graph is False
        assert config.enable_privacy_filtering is False


def test_memory_config_standard():
    """Test creating a standard memory configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = str(Path(temp_dir) / "memory")
        config = MemoryConfig.standard(memory_path=memory_path)

        # Verify standard settings
        assert config.memory_path == memory_path
        assert config.enable_embeddings is True
        assert config.enable_dependency_graph is True
        assert config.enable_privacy_filtering is False


def test_memory_config_full_featured():
    """Test creating a full-featured memory configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = str(Path(temp_dir) / "memory")
        config = MemoryConfig.full_featured(memory_path=memory_path)

        # Verify full-featured settings
        assert config.memory_path == memory_path
        assert config.enable_embeddings is True
        assert config.enable_dependency_graph is True
        assert config.enable_privacy_filtering is True


def test_memory_config_custom_params():
    """Test creating a memory configuration with custom parameters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_path = str(Path(temp_dir) / "memory")
        config = MemoryConfig(
            memory_path=memory_path,
            indexer_type="code",
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            chunk_size=500,
            chunk_overlap=100,
            enable_embeddings=True,
            enable_dependency_graph=False,
        )

        # Verify custom settings
        assert config.memory_path == memory_path
        assert config.indexer_type == "code"
        assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.enable_embeddings is True
        assert config.enable_dependency_graph is False
