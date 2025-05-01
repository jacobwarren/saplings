"""
Tests for the configuration module.
"""

import pytest

from saplings.gasa.config import FallbackStrategy, GASAConfig, MaskStrategy


class TestGASAConfig:
    """Tests for the GASAConfig class."""

    def test_init_default(self):
        """Test initialization with default values."""
        config = GASAConfig()

        assert config.enabled is True
        assert config.max_hops == 2
        assert config.mask_strategy == MaskStrategy.BINARY
        assert config.fallback_strategy == FallbackStrategy.BLOCK_DIAGONAL
        assert config.global_tokens == ["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"]
        assert config.summary_token == "[SUM]"
        assert config.add_summary_token is True
        assert config.block_size == 512
        assert config.overlap == 64
        assert config.soft_mask_temperature == 0.1
        assert config.cache_masks is True
        assert config.cache_dir is None
        assert config.visualize is False
        assert config.visualization_dir is None

    def test_init_custom(self):
        """Test initialization with custom values."""
        config = GASAConfig(
            enabled=False,
            max_hops=3,
            mask_strategy=MaskStrategy.SOFT,
            fallback_strategy=FallbackStrategy.DENSE,
            global_tokens=["[CLS]", "[SEP]"],
            summary_token="[SUMMARY]",
            add_summary_token=False,
            block_size=256,
            overlap=32,
            soft_mask_temperature=0.2,
            cache_masks=False,
            cache_dir="/tmp/cache",
            visualize=True,
            visualization_dir="/tmp/vis",
        )

        assert config.enabled is False
        assert config.max_hops == 3
        assert config.mask_strategy == MaskStrategy.SOFT
        assert config.fallback_strategy == FallbackStrategy.DENSE
        assert config.global_tokens == ["[CLS]", "[SEP]"]
        assert config.summary_token == "[SUMMARY]"
        assert config.add_summary_token is False
        assert config.block_size == 256
        assert config.overlap == 32
        assert config.soft_mask_temperature == 0.2
        assert config.cache_masks is False
        assert config.cache_dir == "/tmp/cache"
        assert config.visualize is True
        assert config.visualization_dir == "/tmp/vis"

    def test_from_cli_args_empty(self):
        """Test from_cli_args method with empty arguments."""
        config = GASAConfig.from_cli_args({})

        assert config.enabled is True
        assert config.max_hops == 2
        assert config.mask_strategy == MaskStrategy.BINARY
        assert config.fallback_strategy == FallbackStrategy.BLOCK_DIAGONAL

    def test_from_cli_args_custom(self):
        """Test from_cli_args method with custom arguments."""
        args = {
            "gasa": False,
            "gasa_hop": 3,
            "gasa_strategy": "soft",
            "gasa_fallback": "dense",
            "gasa_block_size": 256,
            "gasa_overlap": 32,
            "gasa_cache": False,
            "gasa_cache_dir": "/tmp/cache",
            "gasa_visualize": True,
            "gasa_visualization_dir": "/tmp/vis",
        }

        config = GASAConfig.from_cli_args(args)

        assert config.enabled is False
        assert config.max_hops == 3
        assert config.mask_strategy == MaskStrategy.SOFT
        assert config.fallback_strategy == FallbackStrategy.DENSE
        assert config.block_size == 256
        assert config.overlap == 32
        assert config.cache_masks is False
        assert config.cache_dir == "/tmp/cache"
        assert config.visualize is True
        assert config.visualization_dir == "/tmp/vis"
